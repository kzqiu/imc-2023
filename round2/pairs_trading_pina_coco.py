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

class Trader:
    lastAcceptablePrice_pearls = 10000
    lastAcceptablePrice_bananas = 4800
    lastAcceptablePrice_coconuts = 7000
    lastAcceptablePrice_pina_coladas = 14000
    bananas_max = 20
    pearls_max = 20
    pina_coladas_max = 300
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
        orders_coconuts: list[Order] = []
        orders_pina_coladas: list[Order] = []
        coconuts_data = [8000 for i in range(30)]
        pina_coladas_data = [15000 for i in range(30)]

        # Iterate over all the keys (the available products) contained in the order depths
        for product in state.order_depths.keys():

            if product == "COCONUTS":
                order_depth_coconuts: OrderDepth = state.order_depths['COCONUTS']
                order_depth_pina_coladas: OrderDepth = state.order_depths['PINA_COLADAS']

                mid_price_coconuts = (min(order_depth_coconuts.sell_orders.keys()) + max(order_depth_coconuts.buy_orders.keys()))/2
                mid_price_pina_coladas = (min(order_depth_pina_coladas.sell_orders.keys()) + max(order_depth_pina_coladas.buy_orders.keys()))/2

                coconuts_data.append(mid_price_coconuts)
                pina_coladas_data.append(mid_price_pina_coladas)

                d = {'coconuts':coconuts_data, 'pina_coladas':pina_coladas_data}
                df = pd.DataFrame(d)

                spread = df['coconuts'] - df['pina_coladas']

                spread_mean = spread.rolling(window=30).mean()
                spread_std = spread.rolling(window=30).std()
                zscore = (spread - spread_mean) / spread_std

                long_signal = zscore  < -2.0
                short_signal = zscore > 2.0
                exit_signal = abs(zscore) < 1.0



                if long_signal[long_signal.size-1]:
                    #long position
                    #buy pina_coladas
                    if len(order_depth_pina_coladas.sell_orders) > 0:
                        best_ask = min(order_depth_pina_coladas.buy_orders.keys())
                        best_ask_volume = order_depth_pina_coladas.buy_orders[best_ask]
                        print("BUY PINA_COLADAS", str(-best_ask_volume) + "x", best_ask)
                        orders_pina_coladas.append(Order('PINA_COLADAS', best_ask, -best_ask_volume))
                    #sell coconuts
                    if len(order_depth_coconuts.buy_orders) != 0:
                        best_bid = max(order_depth_coconuts.sell_orders.keys())
                        best_bid_volume = order_depth_coconuts.sell_orders[best_bid]
                        print("SELL COCONUTS", str(best_bid_volume) + "x", best_bid)
                        orders_coconuts.append(Order('COCONUTS', best_bid, -best_bid_volume))

                elif short_signal[short_signal.size-1]:
                    #short position
                    #buy coconuts
                    if len(order_depth_coconuts.sell_orders) > 0:
                        best_ask = min(order_depth_coconuts.sell_orders.keys())
                        best_ask_volume = order_depth_coconuts.sell_orders[best_ask]
                        print("BUY COCONUTS", str(-best_ask_volume) + "x", best_ask)
                        orders_coconuts.append(Order('COCONUTS', best_ask, -best_ask_volume))
                    #sell pina_coladas
                    if len(order_depth_pina_coladas.buy_orders) != 0:
                        best_bid = max(order_depth_pina_coladas.buy_orders.keys())
                        best_bid_volume = order_depth_pina_coladas.buy_orders[best_bid]
                        print("SELL PINA_COLADAS", str(best_bid_volume) + "x", best_bid)
                        orders_pina_coladas.append(Order('PINA_COLADAS', best_bid, -best_bid_volume))
                elif exit_signal[exit_signal.size - 1]:
                    # Reset Positions to 0
                    coconut_position = state.position.get('COCONUTS', 0)
                    pina_colada_position = state.position.get('PINA_COLADAS', 0)
                    if coconut_position < 0:
                        coconut_asks = order_depth_coconuts.sell_orders
                        while coconut_position < 0:
                            # Go through dict looking for best sell orders and append order.
                            # Then subtract order amount from coconut_position
                            # Remove order from the dict
                            best_ask = min(coconut_asks.keys())
                            best_ask_volume = order_depth_coconuts.sell_order[best_ask]
                            if best_ask_volume < coconut_position:
                                best_ask_volume = coconut_position
                            order_depth_coconuts.append(Order('COCONUTS', best_ask, -best_ask_volume))
                            coconut_position -= best_ask_volume
                            coconut_asks.pop(best_ask)
                            
                    elif coconut_position > 0:
                        coconut_bids = order_depth_coconuts.buy_orders
                        while coconut_position > 0:
                            # Go through dict looking for best buy orders and append order.
                            # Then subtract order amount from coconut_position
                            # Remove order from the dict
                            best_bid = max(coconut_bids.keys())
                            best_bid_volume = order_depth_coconuts.buy_order[best_bid]
                            if best_bid_volume > coconut_position:
                                best_bid_volume = coconut_position
                            order_depth_coconuts.append(Order('COCONUTS', best_bid, -best_bid_volume))
                            coconut_position -= best_bid_volume
                            coconut_asks.pop(best_bid)

                    if pina_colada_position < 0:
                        pina_colada_asks = order_depth_pina_coladas.sell_orders
                        while pina_colada_position < 0:
                            # Go through dict looking for best sell orders and append order.
                            # Then subtract order amount from pina_colada_position
                            # Remove order from the dict
                            best_ask = min(pina_colada_asks.keys())
                            best_ask_volume = order_depth_pina_coladas.sell_order[best_ask]
                            if best_ask_volume < pina_colada_position:
                                best_ask_volume = pina_colada_position
                            order_depth_pina_coladas.append(Order('PINA_COLADA', best_ask, -best_ask_volume))
                            pina_colada_position -= best_ask_volume
                            pina_colada_asks.pop(best_ask)

                    elif pina_colada_position > 0:
                        pina_colada_bids = order_depth_pina_coladas.sell_orders
                        while pina_colada_position > 0:
                            # Go through dict looking for best sell orders and append order.
                            # Then subtract order amount from pina_colada_position
                            # Remove order from the dict
                            best_bid = min(pina_colada_bids.keys())
                            best_bid_volume = order_depth_pina_coladas.buy_order[best_bid]
                            if best_bid_volume > pina_colada_position:
                                best_bid_volume = pina_colada_position
                            order_depth_pina_coladas.append(Order('PINA_COLADA', best_bid, -best_bid_volume))
                            pina_colada_position -= best_bid_volume
                            pina_colada_bids.pop(best_bid)
                    


        # Add all the above orders to the result dict
        result['COCONUTS'] = orders_coconuts
        result['PINA_COLADAS'] = orders_pina_coladas
        # Return the dict of orders
        # These possibly contain buy or sell orders for PEARLS
        # Depending on the logic above
        logger.flush(state, orders_coconuts + orders_pina_coladas)

        return result