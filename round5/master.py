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
            self.coconuts_data = [8000.0 for i in range(100)]
            self.pina_coladas_data = [15000.0 for i in range(100)]
            self.dolphin_data = [3074.0 for i in range(100)]
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
        orders_coconuts: list[Order] = []
        orders_pina_coladas: list[Order] = []
        orders_berries: list[Order] = []
        orders_pearls: list[Order] = []

        # Iterate over all the keys (the available products) contained in the order depths
        for product in state.order_depths.keys():
            
            if product == 'PEARLS':
                spread = 1
                open_spread = 3
                position_limit = 20
                position_spread = 20
                current_position = state.position.get("PEARLS",0)
                best_ask = 0
                best_bid = 0
                
                order_depth_pearls: OrderDepth = state.order_depths["PEARLS"]
                
                if len(order_depth_pearls.sell_orders) > 0:
                    best_ask = min(order_depth_pearls.sell_orders.keys())

                    if best_ask <= 10000-spread:
                        best_ask_volume = order_depth_pearls.sell_orders[best_ask]
                    else:
                        best_ask_volume = 0
                else:
                    best_ask_volume = 0

                if len(order_depth_pearls.buy_orders) > 0:
                    best_bid = max(order_depth_pearls.buy_orders.keys())

                    if best_bid >= 10000+spread:
                        best_bid_volume = order_depth_pearls.buy_orders[best_bid]
                    else:
                        best_bid_volume = 0 
                else:
                    best_bid_volume = 0

                if current_position - best_ask_volume > position_limit:
                    best_ask_volume = current_position - position_limit
                    open_ask_volume = 0
                else:
                    open_ask_volume = current_position - position_spread - best_ask_volume

                if current_position - best_bid_volume < -position_limit:
                    best_bid_volume = current_position + position_limit
                    open_bid_volume = 0
                else:
                    open_bid_volume = current_position + position_spread - best_bid_volume

                if -open_ask_volume < 0:
                    open_ask_volume = 0         
                if open_bid_volume < 0:
                    open_bid_volume = 0

                if best_ask == 10000-open_spread and -best_ask_volume > 0:
                    print("BUY PEARLS", str(-best_ask_volume-open_ask_volume) + "x", 10000-open_spread)
                    orders_pearls.append(Order("PEARLS", 10000-open_spread, -best_ask_volume-open_ask_volume))
                else:
                    if -best_ask_volume > 0:
                        print("BUY PEARLS", str(-best_ask_volume) + "x", best_ask)
                        orders_pearls.append(Order(product, best_ask, -best_ask_volume))
                    if -open_ask_volume > 0:
                        print("BUY PEARLS", str(-open_ask_volume) + "x", 10000-open_spread)
                        orders_pearls.append(Order(product, 10000-open_spread, -open_ask_volume))

                if best_bid == 10000+open_spread and best_bid_volume > 0:
                    print("SELL PEARLS", str(best_bid_volume+open_bid_volume) + "x", 10000+open_spread)
                    orders_pearls.append(Order(product, 10000+open_spread, -best_bid_volume-open_bid_volume))
                else:
                    if best_bid_volume > 0:
                        print("SELL PEARLS", str(best_bid_volume) + "x", best_bid)
                        orders_pearls.append(Order(product, best_bid, -best_bid_volume))
                    if open_bid_volume > 0:
                        print("SELL PEARLS", str(open_bid_volume) + "x", 10000+open_spread)
                        orders_pearls.append(Order(product, 10000+open_spread, -open_bid_volume))
                        
            
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

                if time >= 90000 and time < 300000 and position != position_limit:
                    best_ask = min(state.order_depths["BERRIES"].sell_orders.items())
                    orders_berries.append(Order("BERRIES", best_ask[0], min(-best_ask[1], position_limit - position)))
                
                if time >= 500000 and time < 505000 and position != -position_limit:
                    best_bid = max(state.order_depths["BERRIES"].buy_orders.items())
                    orders_berries.append(Order("BERRIES", best_bid[0], max(-best_bid[1], - position_limit - position)))

##################################################################################
            
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

##################################################################################
                    


        # Add all the above orders to the result dict
        result['PEARLS'] = orders_pearls
        result['COCONUTS'] = orders_coconuts
        result['PINA_COLADAS'] = orders_pina_coladas
        result['BERRIES'] = orders_berries
        # Return the dict of orders
        # These possibly contain buy or sell orders for PEARLS
        # Depending on the logic above
        logger.flush(state, orders_coconuts + orders_pina_coladas + orders_berries + orders_pearls)

        return result