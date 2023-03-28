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
            self.dolphin_data = [3074 for i in range(100)]
            self.gear_data = [99100.0 for i in range(100)]


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

                order_depth_diving_gear: OrderDepth = state.order_depths['DIVING_GEAR']

                mid_price_diving_gear = (min(order_depth_diving_gear.sell_orders.keys()) + max(order_depth_diving_gear.buy_orders.keys()))/2

                self.dolphin_data.append(dolphins)
                self.gear_data.append(mid_price_diving_gear)
                

                x_positions = state.position.get('DIVING_GEAR', 0)
                
                x = self.gear_data
                y = self.dolphin_data
                
                x_limit = 50
                
                hurst = hurst_exponent(x)
                spread = y - kalman_filter(y, len(x))
                
                upper_bound = np.mean(spread) + hurst * np.std(spread)
                lower_bound = np.mean(spread) - hurst * np.std(spread)
                
                capital = 0
                
                for key, val in order_depth_diving_gear.sell_orders.items():
                    capital += key * val
                    
                signal = 0
                if spread[-1] > upper_bound:
                    # short y, long x
                    if x_positions < x_limit:
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
                    if x_positions < x_limit:
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
                
                diving_gear_position = state.position.get('DIVING_GEAR', 0)

                # Implement risk management using stop-loss orders
                stop_loss = 0.20  # set stop loss at 1%

                if signal == 1:
                    stop_loss_price = np.mean(spread) - stop_loss * np.std(spread)
                    
                    if spread[-1] < stop_loss_price:
                        signal = 0  # close position if stop-loss is triggered
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
                    else:
                        if len(order_depth_diving_gear.buy_orders.keys()) > 0:
                            best_bid = max(order_depth_diving_gear.buy_orders.keys())
                            best_bid_volume = order_depth_diving_gear.buy_orders[best_bid]
                            print("SELL DIVING_GEAR", str(best_bid_volume) + "x", best_bid)
                            orders_diving_gear.append(Order('DIVING_GEAR', best_bid, -best_bid_volume))

                elif signal == -1:
                    stop_loss_price = np.mean(spread) + stop_loss * np.std(spread)
                    
                    if spread[-1] > stop_loss_price:
                        signal = 0  # close position if stop-loss is triggered
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
                    else:
                        if len(order_depth_diving_gear.sell_orders.keys()) > 0:
                            best_ask = min(order_depth_diving_gear.sell_orders.keys())
                            best_ask_volume = order_depth_diving_gear.sell_orders[best_ask]
                            print("BUY DIVING_GEAR", str(-best_ask_volume) + "x", best_ask)
                            orders_diving_gear.append(Order('DIVING_GEAR', best_ask, -best_ask_volume))
                    


        # Add all the above orders to the result dict
        result['DIVING_GEAR'] = orders_diving_gear
        # Return the dict of orders
        # These possibly contain buy or sell orders for PEARLS
        # Depending on the logic above
        logger.flush(state, orders_diving_gear)

        return result