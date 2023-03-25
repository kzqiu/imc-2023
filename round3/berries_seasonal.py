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

class Trader:
    def run(self, state: TradingState) -> dict[Symbol, List[Order]]:
        result = {}
        
        for product in state.order_depths.keys():
            orders: list[Order] = []

##################################################################################

            if product == 'BERRIES':
                position = state.position.get(product, 0)
                position_limit = 250
                time = state.timestamp % 1000000

                """
                Strategy:
                - Buy at start of day + delta (est. 10000)
                - Hold until midday (approx. state.timestamp % 1000000 >= 500000)
                - Hold at midday and switch positions!
                """

                if time >= 100000 and time < 110000 and position != position_limit:
                    best_ask = min(state.order_depths[product].sell_orders.items())
                    orders.append(Order(product, best_ask[0], min(-best_ask[1], position_limit - position)))
                
                if time >= 500000 and time < 510000 and position != -position_limit:
                    best_bid = max(state.order_depths[product].buy_orders.items())
                    orders.append(Order(product, best_bid[0], max(-best_bid[1], - position_limit - position)))

##################################################################################

                result[product] = orders
        
        logger.flush(state, orders) # type: ignore
        return result