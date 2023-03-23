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

"""
ARIMA Functions!
"""
def least_squares(x, y):
    if np.linalg.det(x.T @ x) != 0:
        return np.linalg.inv((x.T @ x)) @ (x.T @ y)
    return np.linalg.pinv((x.T @ x)) @ (x.T @ y) 

# Autoregressor
def ar_process(eps, phi):
    """
    Creates a AR process with a zero mean.
    """
    # Reverse the order of phi and add a 1 for current eps_t
    phi = np.r_[1, phi][::-1]
    ar = eps.copy()
    offset = len(phi)
    for i in range(offset, ar.shape[0]):
        ar[i - 1] = ar[i - offset: i] @ phi
    return ar

# Moving Average 
n = 500
eps = np.random.normal(size=n)

def lag_view(x, order):
    """
    For every value X_i create a row that lags k values: [X_i-1, X_i-2, ... X_i-k]
    """
    y = x.copy()
    # Create features by shifting the window of `order` size by one step.
    # This results in a 2D array [[t1, t2, t3], [t2, t3, t4], ... [t_k-2, t_k-1, t_k]]
    x = np.array([y[-(i + order):][:order] for i in range(y.shape[0])])

    # Reverse the array as we started at the end and remove duplicates.
    # Note that we truncate the features [order -1:] and the labels [order]
    # This is the shifting of the features with one time step compared to the labels
    x = np.stack(x)[::-1][order - 1: -1]
    y = y[order:]

    return x, y

def ma_process(eps, theta):
    """
    Creates an MA(q) process with a zero mean (mean not included in implementation).
    :param eps: (array) White noise signal.
    :param theta: (array/ list) Parameters of the process.
    """
    # reverse the order of theta as Xt, Xt-1, Xt-k in an array is Xt-k, Xt-1, Xt.
    theta = np.array([1] + list(theta))[::-1][:, None]
    eps_q, _ = lag_view(eps, len(theta))
    return eps_q @ theta

# Differencing 
def difference(x, d=1):
    if d == 0:
        return x
    else:
        x = np.r_[x[0], np.diff(x)]
        return difference(x, d - 1)

def undo_difference(x, d=1):
    if d == 1:
        return np.cumsum(x)
    else:
        x = np.cumsum(x)
        return undo_difference(x, d - 1)

# Linear Regression
class LinearModel:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.beta = None
        self.intercept_ = None
        self.coef_ = None

    def _prepare_features(self, x):
        if self.fit_intercept:
            x = np.hstack((np.ones((x.shape[0], 1)), x))
        return x

    def fit(self, x, y):
        x = self._prepare_features(x)
        self.beta = least_squares(x, y)
        if self.fit_intercept:
            self.intercept_ = self.beta[0]
            self.coef_ = self.beta[1:]
        else:
            self.coef_ = self.beta

    def predict(self, x):
        x = self._prepare_features(x)
        return x @ self.beta

    def fit_predict(self, x, y):
        self.fit(x, y)
        return self.predict(x)

# ARIMA Model 
class ARIMA(LinearModel):
    def __init__(self, q, d, p):
        """
        An ARIMA model.
        :param q: (int) Order of the MA model.
        :param p: (int) Order of the AR model.
        :param d: (int) Number of times the data needs to be differenced.
        """
        super().__init__(True)
        self.p = p
        self.d = d
        self.q = q
        self.ar = None
        self.resid = None

    def prepare_features(self, x):
        if self.d > 0:
            x = difference(x, self.d)

        ar_features = None
        ma_features = None

        # Determine the features and the epsilon terms for the MA process
        if self.q > 0:
            if self.ar is None:
                self.ar = ARIMA(0, 0, self.p)
                self.ar.fit_predict(x)
            eps = self.ar.resid
            eps[0] = 0

            # prepend with zeros as there are no residuals_t-k in the first X_t
            ma_features, _ = lag_view(np.r_[np.zeros(self.q), eps], self.q)

        # Determine the features for the AR process
        if self.p > 0:
            # prepend with zeros as there are no X_t-k in the first X_t
            ar_features = lag_view(np.r_[np.zeros(self.p), x], self.p)[0]

        if ar_features is not None and ma_features is not None:
            n = min(len(ar_features), len(ma_features))
            ar_features = ar_features[:n]
            ma_features = ma_features[:n]
            features = np.hstack((ar_features, ma_features))
        elif ma_features is not None:
            n = len(ma_features)
            features = ma_features[:n]
        else:
            n = len(ar_features)
            features = ar_features[:n]

        return features, x[:n]

    def fit(self, x):
        features, x = self.prepare_features(x)
        super().fit(features, x)
        return features

    def fit_predict(self, x):
        """
        Fit and transform input
        :param x: (array) with time series.
        """
        features = self.fit(x)
        return self.predict(x, prepared=(features))

    def predict(self, x, **kwargs):
        """
        :param x: (array)
        :kwargs:
            prepared: (tpl) containing the features, eps and x
        """
        features = kwargs.get('prepared', None)
        if features is None:
            features, x = self.prepare_features(x)

        y = super().predict(features)
        self.resid = x - y

        return self.return_output(y)

    def return_output(self, x):
        if self.d > 0:
            x = undo_difference(x, self.d)
        return x

    def forecast(self, x, n):
        """
        Forecast the time series.
        :param x: (array) Current time steps.
        :param n: (int) Number of time steps in the future.
        """
        features, x = self.prepare_features(x)
        y = super().predict(features)

        # Append n time steps as zeros. Because the epsilon terms are unknown
        y = np.r_[y, np.zeros(n)]
        for i in range(n):
            feat = np.r_[y[-(self.p + n) + i: -n + i], np.zeros(self.q)]
            y[x.shape[0] + i] = super().predict(feat[None, :])
        return self.return_output(y)

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

##################################################################################

# historical data!

class Trader:
    def __init__(self):
        self.holdings = 0
        self.last_trade = 0

        self.history_length = 20
        self.price_history_banana = np.array([])
        self.open_prices = np.array([])
        self.high_prices = np.array([])
        self.low_prices = np.array([]) 
        self.close_prices = np.array([])
        self.model = ARIMA(4, 0, 1)

    def run(self, state: TradingState) -> dict[Symbol, List[Order]]:
        result = {}
        
        for product in state.order_depths.keys():
            orders: list[Order] = []
                
            if product == 'BANANAS':
                # starting at certain timestamp
                enough_data = True
                start_trading = 0

                position_limit = 20
                current_position = state.position.get(product, 0)

                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []
                    
                price = 0
                count = 0
                for Trade in state.market_trades.get(product, []):
                    price += Trade.price * Trade.quantity
                    count += Trade.quantity

                if count == 0:
                    if len(self.price_history_banana) == 0:
                        enough_data = False
                    else:
                        current_avg_market_price = self.price_history_banana[-1]
                else:
                    current_avg_market_price = price / count
           

                if state.timestamp >= start_trading and enough_data == True:
                    # Calculating fair price
                    self.price_history_banana = np.append(self.price_history_banana, current_avg_market_price)
                    pred = self.model.fit_predict(self.price_history_banana)

                    if len(pred) >= self.history_length + 1:
                        pred = pred[1:]

                    forecasted_price = self.model.forecast(pred, 1)[-1]

                    # Calculating spread
                    spread = 4

                    buy_spread = spread / 2
                    sell_spread = spread / 2

                    sell_orders = sorted([sell_ord for sell_ord in order_depth.sell_orders.items()], reverse=True)
                    buy_orders = sorted([buy_ord for buy_ord in order_depth.buy_orders.items()], reverse=True)

                    sell_pos = current_position
                    buy_pos = current_position
                    breakloop = False

                    for order in sell_orders:
                        if sell_pos == 20:
                            break

                        ask = order[0]
                        vol = order[1]

                        if sell_pos - vol > position_limit:
                            vol = sell_pos - position_limit
                            breakloop = True
                        
                        if ask <= forecasted_price - buy_spread and -vol > 0:
                            print("BUY", product, str(-vol) + "x", ask)
                            orders.append(Order(product, ask, -vol))
                            sell_pos += -vol

                        if breakloop:
                            break

                    breakloop = False

                    for order in buy_orders:
                        if current_position == -20:
                            break

                        bid = order[0]
                        vol = order[1]

                        if buy_pos - vol < -position_limit:
                            vol = buy_pos + position_limit
                            breakloop = True

                        if bid >= forecasted_price + sell_spread and vol > 0:
                            print("SELL", product, str(vol) + "x", bid)
                            orders.append(Order(product, bid, -vol))
                            buy_pos -= vol

                        if breakloop:
                            break

                    # add open contracts here! 
                    # check if sell_pos exceeds 20 and buy_pos exceeds -20

            result[product] = orders
        logger.flush(state, orders)
        return result