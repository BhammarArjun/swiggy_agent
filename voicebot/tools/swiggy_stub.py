"""In-memory stub of the Swiggy Food MCP tools. Same function signatures the
real MCP client will expose, so the agent/pipeline never change when swapped.
Every call is logged. Returns canned-but-realistic data."""
import logging

log = logging.getLogger("swiggy_stub")

_ADDRESSES = [
    {"addressId": "addr_home_42", "label": "Home"},
    {"addressId": "addr_work_7", "label": "Work"},
]
_RESTAURANTS = {
    "biryani": [
        {"restaurantId": "rest_paradise_9", "name": "Paradise Biryani", "eta": "35 min"},
        {"restaurantId": "rest_bbq_3", "name": "Biryani House", "eta": "45 min"},
    ],
    "pizza": [
        {"restaurantId": "rest_pizza_1", "name": "Pizza Town", "eta": "30 min"},
    ],
}
_MENUS = {
    "rest_paradise_9": [
        {"itemId": "item_chk_bir", "name": "Chicken Biryani", "price": 320},
        {"itemId": "item_naan", "name": "Butter Naan", "price": 45},
    ],
    "rest_bbq_3": [
        {"itemId": "item_mutton_bir", "name": "Mutton Biryani", "price": 380},
    ],
    "rest_pizza_1": [
        {"itemId": "item_margherita", "name": "Margherita Pizza", "price": 250},
    ],
}

CART_CAP = 1000  # Builders Club cap (rupees)
_cart: dict = {"restaurantId": None, "items": []}
_order_counter = [555]


def reset_state() -> None:
    """Reset cart + order counter (used by tests)."""
    _cart["restaurantId"] = None
    _cart["items"] = []
    _order_counter[0] = 555


def _menu_index(restaurantId: str) -> dict:
    return {m["itemId"]: m for m in _MENUS.get(restaurantId, [])}


def get_addresses() -> list:
    """Retrieve the user's saved delivery addresses."""
    log.info("get_addresses()")
    return _ADDRESSES


def search_restaurants(addressId: str, query: str) -> list:
    """Find restaurants by cuisine or dish near a delivery address.

    Args:
        addressId: A saved address id from get_addresses.
        query: Cuisine or dish to search for, e.g. biryani.
    """
    log.info("search_restaurants(%r, %r)", addressId, query)
    key = next((k for k in _RESTAURANTS if k in query.lower()), "biryani")
    return _RESTAURANTS[key]


def get_restaurant_menu(restaurantId: str) -> list:
    """Get the menu items (itemId, name, price) for a restaurant.

    Args:
        restaurantId: A restaurant id from search_restaurants.
    """
    log.info("get_restaurant_menu(%r)", restaurantId)
    return _MENUS.get(restaurantId, [])


def update_food_cart(restaurantId: str, itemId: str, quantity: int) -> dict:
    """Add an item to the food cart.

    Args:
        restaurantId: The restaurant id the item belongs to.
        itemId: The menu item id from get_restaurant_menu.
        quantity: How many to add.
    """
    log.info("update_food_cart(%r, %r, %s)", restaurantId, itemId, quantity)
    item = _menu_index(restaurantId).get(itemId)
    if item is None:
        return {"ok": False, "error": "unknown item"}
    if _cart["restaurantId"] and _cart["restaurantId"] != restaurantId:
        _cart["items"] = []  # Food cart flushes on restaurant switch
    _cart["restaurantId"] = restaurantId
    _cart["items"].append({"itemId": itemId, "name": item["name"],
                            "price": item["price"], "quantity": int(quantity)})
    return {"ok": True}


def get_food_cart() -> dict:
    """View the current food cart contents and total."""
    log.info("get_food_cart()")
    total = sum(i["price"] * i["quantity"] for i in _cart["items"])
    return {"restaurantId": _cart["restaurantId"], "items": _cart["items"], "total": total}


def apply_food_coupon(couponCode: str, addressId: str) -> dict:
    """Apply a discount coupon code to the cart.

    Args:
        couponCode: The coupon code.
        addressId: The delivery address id.
    """
    log.info("apply_food_coupon(%r, %r)", couponCode, addressId)
    return {"ok": True, "couponCode": couponCode, "discount": 50}


def place_food_order(paymentMethod: str) -> dict:
    """Place the current food order. Only 'COD' (cash on delivery) is supported.

    Args:
        paymentMethod: Must be 'COD'.
    """
    log.info("place_food_order(%r)", paymentMethod)
    if paymentMethod.upper() != "COD":
        return {"success": False, "error": "Only COD supported in v1"}
    _order_counter[0] += 1
    return {"orderId": f"ORD-{_order_counter[0]}", "status": "confirmed"}


# tool list the agent registers
TOOLS = [get_addresses, search_restaurants, get_restaurant_menu,
         update_food_cart, get_food_cart, apply_food_coupon, place_food_order]
