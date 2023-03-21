from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List

class Trader:
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}

        def fair_price(product: str) -> int:
            return 10000

        for product in state.order_depths.keys():
            o_depth: OrderDepth = state.order_depths[product]
            orders: list[Order] = []

            fprice = fair_price(product)
            
            if len(o_depth.sell_orders) != 0:
                best_ask = min(o_depth.sell_orders.keys())
                best_ask_vol = o_depth.sell_orders[best_ask]

                if best_ask < fprice:
                    orders.append(Order(product, best_ask, -best_ask_vol))

            if len(o_depth.buy_orders) != 0:
                best_bid = max(o_depth.buy_orders.keys())
                best_bid_vol = o_depth.buy_orders[best_bid]

                if best_bid > fprice:
                    orders.append(Order(product, best_bid, -best_bid_vol))

            result[product] = orders

        return result
