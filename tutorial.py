from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List

class Trader:
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}

        # moving average fair price
        def moving_avg_fair_price(product: str) -> int:
            if product == 'PEARLS':
                price = 0
                total_vol = 0
                for trade in state.market_trades['PEARLS']:
                    price += trade.price * trade.quantity
                    total_vol += trade.quantity

                return price / count

            if product == 'BANANAS': # Fluctuates! 
                price = 0
                count = 0
                for trade in state.market_trades['PEARLS']:
                    if abs(trade.timestamp) <= 500:
                        price += trade.price * trade.quantity
                        count += trade.quantity

                if len(state.market_trades['PEARLS']) == 0:
                    return 5

                return price / count

            return 10000

        for product in state.order_depths.keys():
            o_depth: OrderDepth = state.order_depths[product]
            orders: list[Order] = []

            fprice = moving_avg_fair_price(product)
            
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
