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

    def run(self, state: TradingState) -> dict[Symbol, List[Order]]:
        result = {}
        
        for product in state.order_depths.keys():
            orders: list[Order] = []
                
            if product == 'BANANAS':
                # current position
                position_limit = 20
                current_position = state.position.get(product, 0)

                buy_orders = state.order_depths[product].buy_orders
                sell_orders = state.order_depths[product].sell_orders
                orders: list[Order] = []

                best_bid = max(buy_orders.keys())
                best_ask = min(sell_orders.keys())

                # Getting fair price, uses last trade by default
                fair_price = self.last_trade

                # If there are valid buy and sell orders, then calculate mean of mid and weighted mid prices
                if len(buy_orders) > 0 and len(sell_orders) > 0:
                    fair_price = (mid_price(best_bid, best_ask) + weighted_mid_price(best_bid, buy_orders[best_bid], best_ask, -sell_orders[best_ask])) / 2
                
                # If fair price is still default value, just skip this iteration!
                if fair_price == 0:
                    continue

                # Getting spread!
                # intervals to look back on for open/high/low/close
                n = 20

                self.open.append(fair_price)
                self.close.append(fair_price)

                if state.timestamp == 0:
                    self.high.append(fair_price)
                    self.low.append(fair_price)
                else:
                    self.high.append(max(self.high[-1], fair_price))
                    self.low.append(max(self.low[-1], fair_price))
                
                if len(self.open) > n:
                    self.high = self.high[1:]
                    self.low = self.low[1:]
                    self.open = self.open[1:]
                    self.close = self.close[1:]

                # Trading code
                if state.timestamp > 1000:
                
                    p_spread = edge(self.open, self.high, self.low, self.close)

                    momentum = slope_momentum(3, self.open)

                    # FULL BID-ASK SPREAD RANGE!
                    spread = p_spread * fair_price

                    # parameter for volume to purchase
                    volume_alpha = 5

                    sells = sorted([it for it in sell_orders.items()], reverse=True)
                    buys = sorted([it for it in buy_orders.items()], reverse=True)

                    agg_vol = current_position

                    for order in sells:
                        if agg_vol == 20:
                            break

                        ask = order[0]
                        vol = order[1]

                        # TODO: Add position component!
                        desired_vol = round(momentum * volume_alpha * spread / (ask - fair_price))
                        real_vol = min(desired_vol, position_limit - agg_vol, -vol)

                        if ask <= fair_price + spread / 2 + momentum:
                            print("BUY", product, str(real_vol) + "x", ask)
                            agg_vol += real_vol 
                            orders.append(Order(product, ask, real_vol))

                    agg_vol = current_position

                    for order in buys:
                        if agg_vol == -20:
                            break

                        bid = order[0]
                        vol = order[1]

                        # TODO: Add position component!
                        desired_vol = round(momentum * volume_alpha * spread / (fair_price - bid))
                        real_vol = max(desired_vol, -position_limit - agg_vol, -vol)

                        if bid >= fair_price - spread / 2 + momentum:
                            print("SELL", product, str(real_vol) + "x", bid)
                            agg_vol += real_vol
                            orders.append(Order(product, bid, real_vol))

            result[product] = orders
        logger.flush(state, orders)
        return result