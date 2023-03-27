import numpy as np
import pandas as pd
import json
from datamodel import Order, OrderDepth, ProsperityEncoder, TradingState, Symbol 
from typing import Any, Dict, List

#################################################################################

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

##################################################################################

class Trader:
    baguette_limit = 150
    dip_limit = 300
    ukulele_limit = 70
    basket_limit = 70
    basket_pnav_ratio = 1.0051
    basket_eps = 0.003
    basket_eps_open = basket_eps * 3

    def __init__(self):
        self.basket_prev: float = None
        self.baguette_prev: float = None
        self.dip_prev: float = None
        self.ukulele_prev: float = None

    def run(self, state: TradingState) -> dict[Symbol, List[Order]]:
        result = {}
        
        for product in state.order_depths.keys():
            orders: list[Order] = []

            if product == 'PICNIC_BASKET':
                # current position
                basket_pos = state.position.get("PICNIC_BASKET", 0)

                basket_buy_orders: List[Order] = state.order_depths[product].buy_orders
                basket_sell_orders: List[Order] = state.order_depths[product].sell_orders

                basket_best_bid: float = max(basket_buy_orders)
                basket_best_ask: float = min(basket_sell_orders)

                # Finding price / NAV ratio
                basket_price: float = (basket_best_bid + basket_best_ask) / 2

                baguette_buy_orders: List[Order] = state.order_depths['BAGUETTE'].buy_orders
                baguette_sell_orders: List[Order] = state.order_depths['BAGUETTE'].sell_orders

                baguette_best_bid: float = max(baguette_buy_orders)
                baguette_best_ask: float = min(baguette_sell_orders)

                baguette_price: float = (baguette_best_bid + baguette_best_ask) / 2
 
                dip_buy_orders: List[Order] = state.order_depths['DIP'].buy_orders
                dip_sell_orders: List[Order] = state.order_depths['DIP'].sell_orders

                dip_best_bid: float = max(dip_buy_orders)
                dip_best_ask: float = min(dip_sell_orders)

                dip_price: float = (dip_best_bid + dip_best_ask) / 2

                ukulele_buy_orders: List[Order] = state.order_depths['UKULELE'].buy_orders
                ukulele_sell_orders: List[Order] = state.order_depths['UKULELE'].sell_orders

                ukulele_best_bid: float = max(ukulele_buy_orders)
                ukulele_best_ask: float = min(ukulele_sell_orders)

                ukulele_price: float = (ukulele_best_bid + ukulele_best_ask) / 2

                est_price: float = 4 * dip_price + 2 * baguette_price + ukulele_price

                price_nav_ratio: float = basket_price / est_price
                print(price_nav_ratio)

                if price_nav_ratio < self.basket_pnav_ratio - self.basket_eps:
                    # ETF is undervalued! -> we buy!
                    basket_best_ask_vol = state.order_depths['PICNIC_BASKET'].sell_orders[basket_best_ask]
                    print("BUY", product, -basket_best_ask_vol, "x", basket_best_ask)
                    orders.append(Order(product, basket_best_ask, -basket_best_ask_vol))
                elif price_nav_ratio > self.basket_pnav_ratio + self.basket_eps:
                    # ETF is overvalued! -> we sell!
                    basket_best_bid_vol = state.order_depths['PICNIC_BASKET'].buy_orders[basket_best_bid]
                    print("SELL", product, -basket_best_bid_vol, "x", basket_best_bid)
                    orders.append(Order(product, basket_best_bid, -basket_best_bid_vol))

                result[product] = orders
        
        logger.flush(state, orders) # type: ignore
        return result