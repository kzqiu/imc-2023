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

#################################################################################

# Mid-price estimators
def mid_price(best_bid: int, best_ask: int) -> float:
    return (best_bid + best_ask) / 2

def weighted_mid_price(best_bid: int, qty_bid: int, best_ask: int, qty_ask: int) -> float:
    imbalance = qty_bid / (qty_bid + qty_ask)

    return imbalance * best_ask + (1 - imbalance) * best_bid

# Edge Spread Estimator!
def edge(open: np.array, high: np.array, low: np.array, close: np.array) -> float:
    """
    Efficient Estimation of Bid-Ask Spreads from OHLC Prices
    Implements an efficient estimation procedure of the bid-ask spread from Open, High, Low, and Close
    prices as proposed in Ardia, Guidotti, Kroencke (2021): https://www.ssrn.com/abstract=3892335
    Prices must be sorted in ascending order of the timestamp.
    :param open: array-like vector of Open prices.
    :param high: array-like vector of High prices.
    :param low: array-like vector of Low prices.
    :param close: array-like vector of Close prices.
    :return: The spread estimate.
    """

    n = len(open)
    if len(high) != n or len(low) != n or len(close) != n:
        raise Exception("open, high, low, close must have the same length")

    o = np.log(np.asarray(open))
    h = np.log(np.asarray(high))
    l = np.log(np.asarray(low))
    c = np.log(np.asarray(close))
    m = (h + l) / 2.

    h1, l1, c1, m1 = h[:-1], l[:-1], c[:-1], m[:-1]
    o, h, l, c, m = o[1:], h[1:], l[1:], c[1:], m[1:]

    x1 = (m - o) * (o - m1) + (m - c1) * (c1 - m1)
    x2 = (m - o) * (o - c1) + (o - c1) * (c1 - m1)

    e1 = np.nanmean(x1)
    e2 = np.nanmean(x2)

    v1 = np.nanvar(x1)
    v2 = np.nanvar(x2)

    if not v1 or not v2:
        return np.nan

    w1 = v2 / (v1 + v2)
    w2 = v1 / (v1 + v2)
    k = 4 * w1 * w2

    n1 = np.nanmean(o == h)
    n2 = np.nanmean(o == l)
    n3 = np.nanmean(c1 == h1)
    n4 = np.nanmean(c1 == l1)
    n5 = np.nanmean(np.logical_and(h == l, l == c1))

    s2 = -4 * (w1 * e1 + w2 * e2) / ((1 - k * (n1 + n2) / 2) + (1 - n5) * (1 - k * (n3 + n4) / 2))
    return float(max(0, s2) ** 0.5)

# slope momentum calculation
def slope_momentum(period: int, mid_prices: np.array) -> float:
    period = min(period, len(mid_prices))
    diffs = np.array(mid_prices[-period + 1:]) - np.array(mid_prices[-period:-1]) # prev

    return np.mean(diffs)
    # return sum([diffs[i] * np.exp(i + 1) for i in range(-1, -period + 1, -1)])

##################################################################################

class Trader:
    def __init__(self):
        self.last_trade = 0

        self.prices = []
        self.open = []
        self.high = []
        self.low = []
        self.close = []
        self.rsi = []
        initial_price = 0

    def run(self, state: TradingState) -> dict[Symbol, List[Order]]:
        result = {}
        
        for product in state.order_depths.keys():
            orders: list[Order] = []
                
            if product == 'BANANAS':
                # current position
                position_limit = 20
                current_position = state.position.get(product, 0)

                # Getting spread!
                # intervals to look back on for open/high/low/close
                spread = 1.5
                open_spread = 2
                start_trading = 0
                position_spread = 15
                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []
                
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())

                if state.timestamp == 0:
                    self.initial_price = mid_price(best_bid, best_ask)

                # Getting fair price, uses last trade by default
                fair_price = state.timestamp * 4.35e-5 + self.initial_price
                    
                if state.timestamp >= start_trading:
                    if len(order_depth.sell_orders) > 0:
                        best_ask = min(order_depth.sell_orders.keys())
                        
                        if best_ask <= fair_price-spread:
                            best_ask_volume = order_depth.sell_orders[best_ask]
                            print("BEST_ASK_VOLUME", best_ask_volume)
                        else:
                            best_ask_volume = 0
                    else:
                        best_ask_volume = 0
                         
                    if len(order_depth.buy_orders) > 0:
                        best_bid = max(order_depth.buy_orders.keys())
                    
                        if best_bid >= fair_price+spread:
                            best_bid_volume = order_depth.buy_orders[best_bid]
                            print("BEST_BID_VOLUME", best_bid_volume)
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

                    if -best_ask_volume > 0:
                        print("BUY", product, str(-best_ask_volume) + "x", best_ask)
                        orders.append(Order(product, best_ask, -best_ask_volume))
                    if -open_ask_volume > 0:
                        print("BUY", product, str(-open_ask_volume) + "x", fair_price-open_spread)
                        orders.append(Order(product, fair_price-open_spread, -open_ask_volume))

                    if best_bid_volume > 0:
                        print("SELL", product, str(best_bid_volume) + "x", best_bid)
                        orders.append(Order(product, best_bid, -best_bid_volume))
                    if open_bid_volume > 0:
                        print("SELL", product, str(open_bid_volume) + "x", fair_price+open_spread)
                        orders.append(Order(product, fair_price+open_spread, -open_bid_volume))
                        
                result[product] = orders

            result[product] = orders
        logger.flush(state, orders)
        return result