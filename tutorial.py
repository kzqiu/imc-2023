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

        for product in state.order_depth.keys():
            # do something
            print(product)

        return result
