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

            if product == 'BERRIES':
                """
                Strategy:
                - Buy at start of day + delta (est. 10000)
                - Hold until midday (approx. state.timestamp % 1000000 >= 500000)
                - Hold at midday and switch positions!
                """

                result[product] = orders
        
        logger.flush(state, orders) # type: ignore
        return result