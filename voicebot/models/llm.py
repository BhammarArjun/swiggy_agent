"""Gemma 4 E2B agent via LiteRT-LM. Pure class, no Pipecat. Holds conversation
state and auto-invokes/chains the registered tools."""
import inspect
import os
from typing import Callable, Iterator

os.environ.setdefault("GLOG_minloglevel", "3")
import litert_lm

from voicebot.config import CONFIG


def _normalize_tool(fn: Callable) -> Callable:
    """Ensure ``inspect.signature(fn)`` reflects the tool's real parameters.

    LiteRT-LM builds each tool's JSON schema from ``inspect.signature``. If a
    tool is a decorator-style wrapper exposing ``(*args, **kwargs)`` (as the
    tests do), the introspected signature is wrong and tool execution fails.
    We rebuild a clean ``__signature__`` from the function's ``__annotations__``
    so the schema (and thus auto-invocation/chaining) works. The original
    callable is still invoked, preserving any side effects (e.g. call tracing).
    """
    sig = inspect.signature(fn)
    has_var = any(
        p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for p in sig.parameters.values()
    )
    if not has_var:
        return fn
    ann = getattr(fn, "__annotations__", {}) or {}
    params = [
        inspect.Parameter(
            name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=annotation,
        )
        for name, annotation in ann.items()
        if name != "return"
    ]
    try:
        fn.__signature__ = inspect.Signature(params)
    except (TypeError, AttributeError):
        pass
    return fn


class GemmaAgent:
    def __init__(self, model_path: str, tools: list[Callable], system_prompt: str):
        self._engine = litert_lm.Engine(
            model_path,
            backend=litert_lm.Backend.CPU(),
            cache_dir=CONFIG.llm_cache_dir,
        )
        self._tools = [_normalize_tool(t) for t in tools]
        self._system_prompt = system_prompt
        self._interrupt = False
        try:
            self._open()
        except Exception:
            self._engine.close()  # don't leak the engine if conversation setup fails
            raise

    def _open(self) -> None:
        msgs = [litert_lm.Message.system(self._system_prompt)]
        self._conv = self._engine.create_conversation(messages=msgs, tools=self._tools)

    def interrupt(self) -> None:
        """Signal the current generation to stop at the next chunk boundary."""
        self._interrupt = True

    def send(self, user_text: str) -> Iterator[str]:
        """Stream assistant text. Tools are auto-invoked/chained by LiteRT-LM."""
        self._interrupt = False
        stream = self._conv.send_message_async(user_text)
        for chunk in stream:
            if self._interrupt:
                break
            for item in chunk.get("content", []):
                if item.get("type") == "text":
                    yield item["text"]

    def reset(self) -> None:
        """Drop conversation history and start fresh."""
        try:
            self._conv.close()
        except Exception:
            pass
        self._open()

    def close(self) -> None:
        try:
            self._conv.close()
        finally:
            self._engine.close()
