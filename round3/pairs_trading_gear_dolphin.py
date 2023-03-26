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
            self.dolphin_data = [3074 for i in range(100)]
            self.gear_data = [99100.0 for i in range(100)]

    lastAcceptablePrice_pearls = 10000
    lastAcceptablePrice_bananas = 4800
    lastAcceptablePrice_coconuts = 7000
    lastAcceptablePrice_pina_coladas = 14000
    bananas_max = 20
    coconuts_max = 600
    diff_from_mean = 0.005


    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """


        # Initialize the method output dict as an empty dict
        result = {}
        # Initialize the list of Orders to be sent as an empty list
        dolphins = state.observations['DOLPHIN_SIGHTINGS']
        orders_diving_gear: list[Order] = []

        # Iterate over all the keys (the available products) contained in the order depths
        for product in state.order_depths.keys():

            if product == "DIVING_GEAR":
                window1 = 5
                window2 = 20
                
                order_depth_diving_gear: OrderDepth = state.order_depths['DIVING_GEAR']

                mid_price_diving_gear = (min(order_depth_diving_gear.sell_orders.keys()) + max(order_depth_diving_gear.buy_orders.keys()))/2

                self.dolphin_data.append(dolphins)
                self.gear_data.append(mid_price_diving_gear)
                

                couple = get_zscore(self.gear_data, self.dolphin_data, window1, window2)
                print(couple)
                zscore = -couple[0]
                ratio = couple[1]
                diving_gear_position = state.position.get('DIVING_GEAR', 0)


                if zscore > 1 and diving_gear_position < 50:
                    if len(order_depth_diving_gear.buy_orders.keys()) > 0:
                        best_bid = max(order_depth_diving_gear.buy_orders.keys())
                        best_bid_volume = order_depth_diving_gear.buy_orders[best_bid]
                        print("SELL DIVING_GEAR", str(best_bid_volume) + "x", best_bid)
                        orders_diving_gear.append(Order('DIVING_GEAR', best_bid, -best_bid_volume))

                elif zscore < -1 and diving_gear_position > -50:
                    #short position
                    #sell pina_coladas
                    if len(order_depth_diving_gear.sell_orders.keys()) > 0:
                        best_ask = min(order_depth_diving_gear.sell_orders.keys())
                        best_ask_volume = order_depth_diving_gear.sell_orders[best_ask]
                        print("BUY DIVING_GEAR", str(-best_ask_volume) + "x", best_ask)
                        orders_diving_gear.append(Order('DIVING_GEAR', best_ask, -best_ask_volume))
                    
                # elif abs(zscore) < 0.5 or (abs(pina_coladas_position) >= 300 and abs(coconuts_position) >= 600):
                elif abs(zscore) < 0.5 or abs(diving_gear_position) > 0:
                    # Reset Positions to 0

                    if diving_gear_position < 0:
                        diving_gear_asks = order_depth_diving_gear.sell_orders
                        while diving_gear_position < 0 and len(diving_gear_asks.keys()) > 0:
                            # Go through dict looking for best sell orders and append order.
                            # Then subtract order amount from pina_colada_position
                            # Remove order from the dict
                            best_ask = min(diving_gear_asks.keys())
                            best_ask_volume = diving_gear_asks[best_ask]
                            if best_ask_volume < diving_gear_position:
                                best_ask_volume = diving_gear_position
                            orders_diving_gear.append(Order('DIVING_GEAR', best_ask, -best_ask_volume))
                            diving_gear_position -= best_ask_volume
                            diving_gear_asks.pop(best_ask)

                    elif diving_gear_position > 0:
                        diving_gear_bids = order_depth_diving_gear.sell_orders
                        while diving_gear_position > 0 and len(diving_gear_bids.keys()) > 0:
                            # Go through dict looking for best sell orders and append order.
                            # Then subtract order amount from pina_colada_position
                            # Remove order from the dict
                            best_bid = min(diving_gear_bids.keys())
                            best_bid_volume = diving_gear_bids[best_bid]
                            if best_bid_volume > diving_gear_position:
                                best_bid_volume = diving_gear_position
                            orders_diving_gear.append(Order('DIVING_GEAR', best_bid, -best_bid_volume))
                            diving_gear_position -= best_bid_volume
                            diving_gear_bids.pop(best_bid)
                    


        # Add all the above orders to the result dict
        result['DIVING_GEAR'] = orders_diving_gear
        # Return the dict of orders
        # These possibly contain buy or sell orders for PEARLS
        # Depending on the logic above
        logger.flush(state, orders_diving_gear)

        return result