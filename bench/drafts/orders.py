import json
import os
from dataclasses import dataclass, field


@dataclass
class Order:
    order_id: str
    customer: str
    items: list = field(default_factory=list)
    shipped: bool = False

    def total(self):
        total = 0.0
        for item in self.items:
            total += item["price"] * item["quantity"]
        return round(total, 2)


def load_orders(path):
    """Read orders from a JSON lines file."""
    orders = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            orders.append(Order(data["order_id"], data["customer"], data.get("items", []), data.get("shipped", False)))
    return orders


def summarize(orders):
    """Count orders and revenue per customer."""
    summary = {}
    for order in orders:
        entry = summary.setdefault(order.customer, {"orders": 0, "revenue": 0.0})
        entry["orders"] += 1
        entry["revenue"] += order.total()
    return summary


def pending(orders):
    return [order for order in orders if not order.shipped]


def main():
    path = os.environ.get("ORDERS", "orders.jsonl")
    orders = load_orders(path)
    summary = summarize(orders)
    for customer, entry in sorted(summary.items()):
        print(f"{customer}: {entry['orders']} orders, {entry['revenue']:.2f} revenue")
    print(f"{len(pending(orders))} orders not shipped yet")


if __name__ == "__main__":
    main()
