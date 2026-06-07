import pytest
from voicebot.tools import swiggy_stub as s


@pytest.fixture(autouse=True)
def reset_cart():
    s.reset_state()
    yield


def test_get_addresses_returns_home():
    addrs = s.get_addresses()
    assert any(a["label"] == "Home" for a in addrs)
    assert all("addressId" in a for a in addrs)


def test_search_then_menu_then_cart_flow():
    addr = s.get_addresses()[0]["addressId"]
    rests = s.search_restaurants(addr, "biryani")
    assert rests and "restaurantId" in rests[0]
    rid = rests[0]["restaurantId"]
    menu = s.get_restaurant_menu(rid)
    item = menu[0]
    res = s.update_food_cart(rid, item["itemId"], 2)
    assert res["ok"] is True
    cart = s.get_food_cart()
    assert cart["items"][0]["quantity"] == 2
    assert cart["total"] == item["price"] * 2


def test_place_order_requires_cod_and_returns_orderid():
    addr = s.get_addresses()[0]["addressId"]
    rid = s.search_restaurants(addr, "biryani")[0]["restaurantId"]
    item = s.get_restaurant_menu(rid)[0]
    s.update_food_cart(rid, item["itemId"], 1)
    res = s.place_food_order("COD")
    assert res["status"] == "confirmed"
    assert res["orderId"].startswith("ORD-")


def test_place_order_rejects_non_cod():
    res = s.place_food_order("CARD")
    assert res.get("success") is False
