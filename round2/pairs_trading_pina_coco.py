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

def get_zscore(S1, S2, window1, window2):
    S1 = pd.DataFrame(S1)
    S2 = pd.DataFrame(S2)
    
    # If window length is 0, algorithm doesn't make sense, so exit
    if (window1 == 0) or (window2 == 0):
        return 0

    # Compute rolling mean and rolling standard deviation
    ratios = S1/S2
    ma1 = ratios[-window1:].mean()
    ma2 = ratios[-window2:].mean()
    std = ratios[-window1:].std()
    zscore = (ma1 - ma2)/std

    return [zscore[0], float(ratios.iloc[-1])]

class Trader:
    def __init__(self):
            self.coconuts_data = [8000.0 for i in range(200)]
            self.pina_coladas_data = [15000.0 for i in range(200)]


    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """


        # Initialize the method output dict as an empty dict
        result = {}
        # Initialize the list of Orders to be sent as an empty list
        orders_coconuts: list[Order] = []
        orders_pina_coladas: list[Order] = []

        # Iterate over all the keys (the available products) contained in the order depths
        for product in state.order_depths.keys():

            if product == "COCONUTS":
                window1 = 120
                window2 = 5
                
                order_depth_coconuts: OrderDepth = state.order_depths['COCONUTS']
                order_depth_pina_coladas: OrderDepth = state.order_depths['PINA_COLADAS']

                mid_price_coconuts = (min(order_depth_coconuts.sell_orders.keys()) + max(order_depth_coconuts.buy_orders.keys()))/2
                mid_price_pina_coladas = (min(order_depth_pina_coladas.sell_orders.keys()) + max(order_depth_pina_coladas.buy_orders.keys()))/2

                self.coconuts_data.append(mid_price_coconuts)
                self.pina_coladas_data.append(mid_price_pina_coladas)
                

                couple = get_zscore(self.pina_coladas_data, self.coconuts_data, window1, window2)
                print(couple)
                zscore = couple[0]
                ratio = couple[1]
                coconuts_position = state.position.get('COCONUTS', 0)
                pina_coladas_position = state.position.get('PINA_COLADAS', 0)


                if zscore > 1 and pina_coladas_position < 300 and coconuts_position > -600:
                    if len(order_depth_pina_coladas.buy_orders.keys()) > 0:
                        best_bid = max(order_depth_pina_coladas.buy_orders.keys())
                        best_bid_volume = order_depth_pina_coladas.buy_orders[best_bid]
                        print("SELL PINA_COLADAS", str(best_bid_volume) + "x", best_bid)
                        orders_pina_coladas.append(Order('PINA_COLADAS', best_bid, -best_bid_volume))
                    #sell coconuts
                    if len(order_depth_coconuts.buy_orders.keys()) > 0:
                        best_bid = max(order_depth_coconuts.buy_orders.keys())
                        best_bid_volume = order_depth_coconuts.buy_orders[best_bid]
                        print("SELL COCONUTS", str(best_bid_volume) + "x", best_bid)
                        orders_coconuts.append(Order('COCONUTS', best_bid, -best_bid_volume*ratio))

                elif zscore < -1 and pina_coladas_position > -300 and coconuts_position < 600:
                    #short position
                    #buy coconuts
                    if len(order_depth_coconuts.sell_orders.keys()) > 0:
                        best_ask = min(order_depth_coconuts.sell_orders.keys())
                        best_ask_volume = order_depth_coconuts.sell_orders[best_ask]
                        print("BUY COCONUTS", str(-best_ask_volume) + "x", best_ask)
                        orders_coconuts.append(Order('COCONUTS', best_ask, -best_ask_volume*ratio))
                    #sell pina_coladas
                    if len(order_depth_pina_coladas.sell_orders.keys()) > 0:
                        best_ask = min(order_depth_pina_coladas.sell_orders.keys())
                        best_ask_volume = order_depth_pina_coladas.sell_orders[best_ask]
                        print("BUY PINA_COLADAS", str(-best_ask_volume) + "x", best_ask)
                        orders_pina_coladas.append(Order('PINA_COLADAS', best_ask, -best_ask_volume))
                    
                elif abs(zscore) < 0.5 or (abs(pina_coladas_position) >= 300 and abs(coconuts_position) >= 600):
                # elif abs(zscore) < 0.5:
                    # Reset Positions to 0
                    if coconuts_position < 0:
                        coconut_asks = order_depth_coconuts.sell_orders
                        while coconuts_position < 0 and len(coconut_asks.keys()) > 0:
                            # Go through dict looking for best sell orders and append order.
                            # Then subtract order amount from coconut_position
                            # Remove order from the dict
                            best_ask = min(coconut_asks.keys())
                            best_ask_volume = coconut_asks[best_ask]
                            if best_ask_volume < coconuts_position:
                                best_ask_volume = coconuts_position
                            orders_coconuts.append(Order('COCONUTS', best_ask, -best_ask_volume))
                            coconuts_position -= best_ask_volume
                            coconut_asks.pop(best_ask)
                            
                    elif coconuts_position > 0:
                        coconut_bids = order_depth_coconuts.buy_orders
                        while coconuts_position > 0 and len(coconut_bids.keys()) > 0:
                            # Go through dict looking for best buy orders and append order.
                            # Then subtract order amount from coconut_position
                            # Remove order from the dict
                            best_bid = max(coconut_bids.keys())
                            best_bid_volume = coconut_bids[best_bid]
                            if best_bid_volume > coconuts_position:
                                best_bid_volume = coconuts_position
                            orders_coconuts.append(Order('COCONUTS', best_bid, -best_bid_volume))
                            coconuts_position -= best_bid_volume
                            coconut_bids.pop(best_bid)

                    if pina_coladas_position < 0:
                        pina_colada_asks = order_depth_pina_coladas.sell_orders
                        while pina_coladas_position < 0 and len(pina_colada_asks.keys()) > 0:
                            # Go through dict looking for best sell orders and append order.
                            # Then subtract order amount from pina_colada_position
                            # Remove order from the dict
                            best_ask = min(pina_colada_asks.keys())
                            best_ask_volume = pina_colada_asks[best_ask]
                            if best_ask_volume < pina_coladas_position:
                                best_ask_volume = pina_coladas_position
                            orders_pina_coladas.append(Order('PINA_COLADAS', best_ask, -best_ask_volume))
                            pina_coladas_position -= best_ask_volume
                            pina_colada_asks.pop(best_ask)

                    elif pina_coladas_position > 0:
                        pina_colada_bids = order_depth_pina_coladas.sell_orders
                        while pina_coladas_position > 0 and len(pina_colada_bids.keys()) > 0:
                            # Go through dict looking for best sell orders and append order.
                            # Then subtract order amount from pina_colada_position
                            # Remove order from the dict
                            best_bid = min(pina_colada_bids.keys())
                            best_bid_volume = pina_colada_bids[best_bid]
                            if best_bid_volume > pina_coladas_position:
                                best_bid_volume = pina_coladas_position
                            orders_pina_coladas.append(Order('PINA_COLADAS', best_bid, -best_bid_volume))
                            pina_coladas_position -= best_bid_volume
                            pina_colada_bids.pop(best_bid)
                    


        # Add all the above orders to the result dict
        result['COCONUTS'] = orders_coconuts
        result['PINA_COLADAS'] = orders_pina_coladas
        # Return the dict of orders
        # These possibly contain buy or sell orders for PEARLS
        # Depending on the logic above
        logger.flush(state, orders_coconuts + orders_pina_coladas)

        return result