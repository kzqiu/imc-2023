from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List

class Trader:
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Takes all buy and sell orders for all symbols as an input,
        and outputs list of orders to be sent.
        """
        result = {}

        for product in state.order_depth.keys():
            # do something
            print(product)

        return result
