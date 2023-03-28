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

def hurst_exponent(series):
    lags = range(2, 100)
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def kalman_filter(y, n):
    # Initialize Kalman filter parameters
    xhat = np.zeros(n)
    P = np.zeros(n)
    xhatminus = np.zeros(n)
    Pminus = np.zeros(n)
    K = np.zeros(n)

    # Set initial values
    xhat[0] = y[0]
    P[0] = 1.0

    # Define measurement and process noises
    R = 0.01
    Q = 0.00001

    # Run Kalman filter
    for k in range(1, n):
        # Time update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1] + Q
        
        # Measurement update
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (y[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]
        
    return xhat

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
                
                order_depth_coconuts: OrderDepth = state.order_depths['COCONUTS']
                order_depth_pina_coladas: OrderDepth = state.order_depths['PINA_COLADAS']

                mid_price_coconuts = (min(order_depth_coconuts.sell_orders.keys()) + max(order_depth_coconuts.buy_orders.keys()))/2
                mid_price_pina_coladas = (min(order_depth_pina_coladas.sell_orders.keys()) + max(order_depth_pina_coladas.buy_orders.keys()))/2
                
                x_positions = state.position.get('COCONUTS', 0)
                y_positions = state.position.get('PINA_COLADAS', 0)

                self.coconuts_data.append(mid_price_coconuts)
                self.pina_coladas_data.append(mid_price_pina_coladas)
                
                x = self.coconuts_data
                y = self.pina_coladas_data
                
                x_limit = 300
                y_limit = 600

                hurst = hurst_exponent(x)
                spread = y - kalman_filter(y, len(x))
                
                upper_bound = np.mean(spread) + hurst * np.std(spread)
                lower_bound = np.mean(spread) - hurst * np.std(spread)
                
                capital = 0
                
                for key, val in order_depth_coconuts.sell_orders.items():
                    capital += key * val
                    
                for key, val in order_depth_pina_coladas.sell_orders.items():
                    capital += key * val
                
                signal = 0
                if spread[-1] > upper_bound:
                    # short y, long x
                    if y_positions < y_limit and x_positions < x_limit:
                        # Calculate position size based on Kelly criterion
                        expected_return = np.mean(spread) - spread[-1]
                        risk = np.std(spread)
                        kelly_fraction = expected_return / (risk ** 2)
                        position_size = kelly_fraction * capital

                        # Adjust position size based on volatility
                        position_size = np.ceil(position_size / (2 * np.std(spread)))

                        signal = -1
                    else:
                        signal = 0  # position limits reached
                elif spread[-1] < lower_bound:
                    # long y, short x
                    if y_positions < y_limit and x_positions < x_limit:
                        # Calculate position size based on Kelly criterion
                        expected_return = spread[-1] - np.mean(spread)
                        risk = np.std(spread)
                        kelly_fraction = expected_return / (risk ** 2)
                        position_size = kelly_fraction * capital

                        # Adjust position size based on volatility
                        position_size = np.ceil(position_size / (2 * np.std(spread)))

                        signal = 1
                    else:
                        signal = 0  # position limits reached
                else:
                    signal = 0  # no position
                    
                coconuts_position = state.position.get('COCONUTS', 0)
                pina_coladas_position = state.position.get('PINA_COLADAS', 0)

                # Implement risk management using stop-loss orders
                stop_loss = 0.60  # set stop loss at 1%

                # if signal == -1 and pina_coladas_position > -300 and coconuts_position < 600:
                if signal == -1:
                    stop_loss_price = np.mean(spread) + stop_loss * np.std(spread)
                    
                    if spread[-1] > stop_loss_price:
                        signal = 0  # close position if stop-loss is triggered
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
                    else:
                        #sell pina_coladas
                        if len(order_depth_pina_coladas.buy_orders.keys()) > 0:
                            best_bid = max(order_depth_pina_coladas.buy_orders.keys())
                            best_bid_volume = order_depth_pina_coladas.buy_orders[best_bid]
                            print("SELL PINA_COLADAS", str(best_bid_volume) + "x", best_bid)
                            orders_pina_coladas.append(Order('PINA_COLADAS', best_bid, -best_bid_volume))
                        #buy coconuts
                        if len(order_depth_coconuts.sell_orders.keys()) > 0:
                            best_ask = min(order_depth_coconuts.sell_orders.keys())
                            best_ask_volume = order_depth_coconuts.sell_orders[best_ask]
                            print("BUY COCONUTS", str(-best_ask_volume) + "x", best_ask)
                            orders_coconuts.append(Order('COCONUTS', best_ask, -best_ask_volume))
                    

                # elif signal == 1 and pina_coladas_position < 300 and coconuts_position > -600:
                elif signal == 1:
                    stop_loss_price = np.mean(spread) - stop_loss * np.std(spread)
                    if spread[-1] < stop_loss_price:
                        signal = 0  # close position if stop-loss is triggered
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
                    #short position
                    else:
                        #sell coconuts
                        if len(order_depth_coconuts.buy_orders.keys()) > 0:
                            best_bid = max(order_depth_coconuts.buy_orders.keys())
                            best_bid_volume = order_depth_coconuts.buy_orders[best_bid]
                            print("SELL COCONUTS", str(best_bid_volume) + "x", best_bid)
                            orders_coconuts.append(Order('COCONUTS', best_bid, -best_bid_volume))
                        #sell pina_coladas
                        if len(order_depth_pina_coladas.sell_orders.keys()) > 0:
                            best_ask = min(order_depth_pina_coladas.sell_orders.keys())
                            best_ask_volume = order_depth_pina_coladas.sell_orders[best_ask]
                            print("BUY PINA_COLADAS", str(-best_ask_volume) + "x", best_ask)
                            orders_pina_coladas.append(Order('PINA_COLADAS', best_ask, -best_ask_volume))
                    
                # elif signal == 0 or (abs(pina_coladas_position) >= 300 and abs(coconuts_position) >= 600):
                    


        # Add all the above orders to the result dict
        result['COCONUTS'] = orders_coconuts
        result['PINA_COLADAS'] = orders_pina_coladas
        # Return the dict of orders
        # These possibly contain buy or sell orders for PEARLS
        # Depending on the logic above
        logger.flush(state, orders_coconuts + orders_pina_coladas)

        return result