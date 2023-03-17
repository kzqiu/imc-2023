from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List

class Trader:
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        1. Given a TradingState Object in each iteration
            - Contains all trades since last iteration
            - Provides list of outstanding quotes from bots (per product)
        2. run method sends out orders to match quotes
            - If algorithm sends a buy price >= than bot quotes,
              there will be trade.
            - If algorithm sends buy order with greater qunatity than bot sell quotes
              the remaning quantity will be left as an outstanding buy quote with
              which bots will potentially continue trading with.
        3. In next iteration, TradingState will reveal whether bots traded on
           outstanding quote. If not, then quote is cancelled at end of the iteration.
        """

        """
        Reducing risk of exceeding position limits
        - Look at position limits vs current positions
        - Add/subtract instant trades (automatically matched prices)
        - Only add additional quotes outside of immediate trades to remain within limits
          as they expire anyways if no bots trade on outstanding player quote.
        """
        
        """
        Takes all buy and sell orders for all symbols as an input,
        and outputs list of orders to be sent.
        """
        result = {}

        def get_fair_price(product: str) -> int:
            if product == 'PEARLS': # Supposed to be pretty stable, Take avg. of historical prices
                price = 0
                count = 0
                for trade in state.market_trades['PEARLS']:
                    price += trade.price * trade.quantity
                    count += trade.quantity

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

            fprice = get_fair_price(product)
            
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
