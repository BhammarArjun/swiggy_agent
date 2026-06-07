import os
import pytest
from voicebot.config import CONFIG
from voicebot.tools import swiggy_stub
from voicebot.prompts.system import SWIGGY_SYSTEM_PROMPT
from voicebot.models.llm import GemmaAgent

pytestmark = pytest.mark.skipif(
    not os.path.exists(CONFIG.llm_model_path),
    reason="LiteRT-LM model not present",
)


@pytest.fixture(autouse=True)
def reset_cart():
    swiggy_stub.reset_state()
    yield


def test_agent_chains_tools_to_build_cart():
    calls = []
    def wrap(fn):
        def inner(*a, **k):
            calls.append(fn.__name__)
            return fn(*a, **k)
        inner.__name__ = fn.__name__
        inner.__doc__ = fn.__doc__
        inner.__annotations__ = fn.__annotations__
        return inner
    tools = [wrap(f) for f in swiggy_stub.TOOLS]

    agent = GemmaAgent(CONFIG.llm_model_path, tools, SWIGGY_SYSTEM_PROMPT)
    reply = "".join(agent.send(
        "Add 2 chicken biryanis to my cart from a biryani place near my home. "
        "Find my address, the restaurant and the item ids yourself."))

    assert "get_addresses" in calls
    assert "search_restaurants" in calls
    assert "get_restaurant_menu" in calls
    assert "update_food_cart" in calls
    assert calls.index("get_addresses") < calls.index("search_restaurants") < \
           calls.index("get_restaurant_menu") < calls.index("update_food_cart")

    cart = swiggy_stub.get_food_cart()
    assert cart["items"], "cart should not be empty"
    assert cart["items"][0]["itemId"] == "item_chk_bir"
    assert isinstance(reply, str) and reply.strip()


def test_agent_does_not_place_order_without_confirmation():
    calls = []
    def wrap(fn):
        def inner(*a, **k):
            calls.append(fn.__name__)
            return fn(*a, **k)
        inner.__name__ = fn.__name__
        inner.__doc__ = fn.__doc__
        inner.__annotations__ = fn.__annotations__
        return inner
    tools = [wrap(f) for f in swiggy_stub.TOOLS]

    agent = GemmaAgent(CONFIG.llm_model_path, tools, SWIGGY_SYSTEM_PROMPT)
    "".join(agent.send("I want a chicken biryani from a place near home."))
    # The agent may browse (search/menu/cart) but must NOT place the order
    # before the user explicitly confirms.
    assert "place_food_order" not in calls
