import numpy as np
import pandas as pd
import json
from datamodel import Order, OrderDepth, ProsperityEncoder, TradingState, Symbol 
from typing import Any, Dict, List


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

    def run(self, state: TradingState) -> dict[Symbol, List[Order]]:

        result = {}
        orders_pearls: list[Order] = []
        orders_bananas: list[Order] = []
        orders_picnic_basket: list[Order] = []
        
        for product in state.order_depths.keys():
            
            if product == 'PEARLS':
                spread = 1
                open_spread = 3
                position_limit = 20
                position_spread = 20
                current_position = state.position.get("PEARLS",0)
                best_ask = 0
                best_bid = 0
                
                order_depth_pearls: OrderDepth = state.order_depths["PEARLS"]
                
                if len(order_depth_pearls.sell_orders) > 0:
                    best_ask = min(order_depth_pearls.sell_orders.keys())

                    if best_ask <= 10000-spread:
                        best_ask_volume = order_depth_pearls.sell_orders[best_ask]
                    else:
                        best_ask_volume = 0
                else:
                    best_ask_volume = 0

                if len(order_depth_pearls.buy_orders) > 0:
                    best_bid = max(order_depth_pearls.buy_orders.keys())

                    if best_bid >= 10000+spread:
                        best_bid_volume = order_depth_pearls.buy_orders[best_bid]
                    else:
                        best_bid_volume = 0 
                else:
                    best_bid_volume = 0

                if current_position - best_ask_volume > position_limit:
                    best_ask_volume = current_position - position_limit
                    open_ask_volume = 0
                else:
                    open_ask_volume = current_position - position_spread - best_ask_volume

                if current_position - best_bid_volume < -position_limit:
                    best_bid_volume = current_position + position_limit
                    open_bid_volume = 0
                else:
                    open_bid_volume = current_position + position_spread - best_bid_volume

                if -open_ask_volume < 0:
                    open_ask_volume = 0         
                if open_bid_volume < 0:
                    open_bid_volume = 0

                if best_ask == 10000-open_spread and -best_ask_volume > 0:
                    print("BUY PEARLS", str(-best_ask_volume-open_ask_volume) + "x", 10000-open_spread)
                    orders_pearls.append(Order("PEARLS", 10000-open_spread, -best_ask_volume-open_ask_volume))
                else:
                    if -best_ask_volume > 0:
                        print("BUY PEARLS", str(-best_ask_volume) + "x", best_ask)
                        orders_pearls.append(Order(product, best_ask, -best_ask_volume))
                    if -open_ask_volume > 0:
                        print("BUY PEARLS", str(-open_ask_volume) + "x", 10000-open_spread)
                        orders_pearls.append(Order(product, 10000-open_spread, -open_ask_volume))

                if best_bid == 10000+open_spread and best_bid_volume > 0:
                    print("SELL PEARLS", str(best_bid_volume+open_bid_volume) + "x", 10000+open_spread)
                    orders_pearls.append(Order(product, 10000+open_spread, -best_bid_volume-open_bid_volume))
                else:
                    if best_bid_volume > 0:
                        print("SELL PEARLS", str(best_bid_volume) + "x", best_bid)
                        orders_pearls.append(Order(product, best_bid, -best_bid_volume))
                    if open_bid_volume > 0:
                        print("SELL PEARLS", str(open_bid_volume) + "x", 10000+open_spread)
                        orders_pearls.append(Order(product, 10000+open_spread, -open_bid_volume))
                        
                result[product] = orders_pearls
                
                
            if product == 'BANANAS':
                position_limit = 20
                spread = 1
                current_position = state.position.get(product, 0)
                        
                order_depth_bananas: OrderDepth = state.order_depths[product]
                
                #Find the worst bid and ask, then buys and sells at those prices
                #If those orders exist, it's because someone is actually buying/selling them (Pablo)
                best_ask = min(order_depth_bananas.buy_orders.keys())
                best_ask_volume = -order_depth_bananas.buy_orders[best_ask]
                best_bid = max(order_depth_bananas.sell_orders.keys())
                best_bid_volume = -order_depth_bananas.sell_orders[best_bid]
            
                if current_position - best_ask_volume > position_limit:
                    best_ask_volume = current_position - position_limit
                if current_position - best_bid_volume < -position_limit:
                    best_bid_volume = current_position + position_limit
            
                print("BUY BANANAS", str(-best_ask_volume) + "x", best_ask+spread)
                orders_bananas .append(Order(product, best_ask+spread, -best_ask_volume))
                print("SELL BANANAS", str(best_bid_volume) + "x", best_bid-spread)
                orders_bananas .append(Order(product, best_bid-spread, -best_bid_volume))
                
                result[product] = orders_bananas                
                
            if product == 'PICNIC_BASKET':
                position_limit = 70
                spread = 3
                current_position = state.position.get(product, 0)
                        
                order_depth_picnic_basket: OrderDepth = state.order_depths[product]
    
                best_ask = min(order_depth_picnic_basket.buy_orders.keys())
                best_ask_volume = -order_depth_picnic_basket.buy_orders[best_ask]
                best_bid = max(order_depth_picnic_basket.sell_orders.keys())
                best_bid_volume = -order_depth_picnic_basket.sell_orders[best_bid]
            
                if current_position - best_ask_volume > position_limit:
                    best_ask_volume = current_position - position_limit
                if current_position - best_bid_volume < -position_limit:
                    best_bid_volume = current_position + position_limit
            
                print("BUY PICNIC BASKET", str(-best_ask_volume) + "x", best_ask+spread)
                orders_picnic_basket.append(Order(product, best_ask+spread, -best_ask_volume))
                print("SELL PICNIC BASKET", str(best_bid_volume) + "x", best_bid-spread)
                orders_picnic_basket.append(Order(product, best_bid-spread, -best_bid_volume))
                
                result[product] = orders_picnic_basket                
                
                
        logger.flush(state, orders_pearls + orders_bananas + orders_picnic_basket)
        return result

