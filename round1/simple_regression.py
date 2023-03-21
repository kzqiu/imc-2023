import numpy as np
import pandas as pd
import json
from datamodel import Order, OrderDepth, ProsperityEncoder, TradingState, Symbol
from typing import Any, Dict, List

def linear(t):
    return -4.29954784333283e-05*t + 4994.36921787607

class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]]) -> None:
        logs = self.logs
        if logs.endswith("\n"):
            logs = logs[:-1]

        print(json.dumps({
            "state": state,
            "orders": orders,
            "logs": logs,
        }, cls=ProsperityEncoder, separators=(",", ":"), sort_keys=True))

        self.state = None
        self.orders = {}
        self.logs = ""

logger = Logger()


"""
Executes the trades
"""
class Trader:
    def __init__(self):
        self.holdings = 0
        self.last_trade = 0

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        
        result = {}
        
        for product in state.order_depths.keys():
            if product == 'BANANAS':

                std_dev = 0.11894678005576327
                    
                order_depth: OrderDepth = state.order_depths[product]

                orders: list[Order] = []

                rate = 1

                lower_curr = linear(state.timestamp) - std_dev*rate
                upper_curr = linear(state.timestamp) + std_dev*rate
                

                if len(order_depth.sell_orders) > 0:

                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_volume = order_depth.sell_orders[best_ask]

                    if best_ask <= lower_curr and np.abs(best_ask_volume) > 0:
                        print("BUY", product, str(-best_ask_volume) + "x", best_ask)
                        orders.append(Order(product, best_ask, -best_ask_volume))

                if len(order_depth.buy_orders) != 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_volume = order_depth.buy_orders[best_bid]
                    
                    if best_bid >= upper_curr and best_bid_volume > 0:
                        print("SELL", product, str(best_bid_volume) + "x", best_bid)
                        orders.append(Order(product, best_bid, -best_bid_volume))

                        
                        
                result[product] = orders
                
                
                
            # if product == 'BANANAS':
                
            #     start_trading = 2000
            #     position_limit = 20
            #     current_position = state.position.get(product,0)
            #     history_length = 20
            #     spread = 3
                
            #     order_depth: OrderDepth = state.order_depths[product]

            #     price = 0
            #     count = 0.000001

            #     for Trade in state.market_trades.get(product, []):
            #         price += Trade.price * Trade.quantity
            #         count += Trade.quantity
            #     current_avg_market_price = price / count
                
            #     price_history_banana = np.append(price_history_banana, current_avg_market_price)
            #     if len(price_history_banana) == history_length+1:
            #         price_history_banana = price_history_banana[1:]
                
            #     orders: list[Order] = []

            #     if state.timestamp >= start_trading:

            #         df_banana_prices = pd.DataFrame(price_history_banana, columns=['mid_price'])
                    
            #         sma = get_sma(df_banana_prices, rate)[-1]
            #         std = get_std(df_banana_prices, rate)[-1]

            #         upper = sma + m * std
            #         lower = sma - m * std

            #         if len(order_depth.sell_orders) > 0:
            #             best_ask = min(order_depth.sell_orders.keys())
                        
            #             if best_ask <= lower:
            #                 best_ask_volume = order_depth.sell_orders[best_ask]
            #             else:
            #                 best_ask_volume = 0
            #         else:
            #             best_ask_volume = 0
                         
            #         if len(order_depth.buy_orders) > 0:
            #             best_bid = max(order_depth.buy_orders.keys())
                    
            #             if best_bid >= upper:
            #                 best_bid_volume = order_depth.buy_orders[best_bid]
            #             else:
            #                 best_bid_volume = 0 
            #         else:
            #             best_bid_volume = 0
                
            #         if current_position - best_ask_volume - best_bid_volume > position_limit:
            #             best_ask_volume = current_position - best_bid_volume - position_limit
            #         elif current_position - best_ask_volume - best_bid_volume < -position_limit:
            #             best_bid_volume = current_position + position_limit - best_ask_volume
                        
            #         position_spread = 15
            #         open_ask_volume = current_position - position_spread - best_ask_volume
            #         open_bid_volume = current_position + position_spread - best_bid_volume
                        
            #         if best_ask == lower and -best_ask_volume > 0:
            #             print("BUY", product, str(-best_ask_volume-open_ask_volume) + "x", lower)
            #             orders.append(Order(product, lower, -best_ask_volume-open_ask_volume))
            #         else:
            #             if -best_ask_volume > 0:
            #                 print("BUY", product, str(-best_ask_volume) + "x", best_ask)
            #                 orders.append(Order(product, best_ask, -best_ask_volume))
            #             if -open_ask_volume > 0:
            #                 print("BUY", product, str(-open_ask_volume) + "x", lower)
            #                 orders.append(Order(product, lower, -open_ask_volume))
                        
            #         if best_bid == upper and best_bid_volume > 0:
            #             print("SELL", product, str(best_bid_volume+open_bid_volume) + "x", upper)
            #             orders.append(Order(product, upper, best_bid_volume+open_bid_volume))
            #         else:
            #             if best_bid_volume > 0:
            #                 print("SELL", product, str(best_bid_volume) + "x", best_bid)
            #                 orders.append(Order(product, best_bid, -best_bid_volume))
            #             if open_bid_volume > 0:
            #                 print("SELL", product, str(open_bid_volume) + "x", upper)
            #                 orders.append(Order(product, lower, -open_bid_volume))
            #     result[product] = orders

            logger.flush(state, orders)

        return result
