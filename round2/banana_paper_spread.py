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



def least_squares(x, y):
    if np.linalg.det(x.T @ x) != 0:
        return np.linalg.inv((x.T @ x)) @ (x.T @ y)
    return np.linalg.pinv((x.T @ x)) @ (x.T @ y) 
"""
Autoregressor
"""
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


"""
Moving Average 
"""
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


"""
Differencing 
"""
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


"""
Linear Regression
"""
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


"""
ARIMA Model 
"""
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
    
mid_price = np.array([])
price_history_banana = np.array([])
open_p = np.array([])
high_p = np.array([])
low_p = np.array([])
close_p = np.array([])

"""
Executes the trades
"""
class Trader:
    def __init__(self):
        self.holdings = 0
        self.last_trade = 0

    def run(self, state: TradingState) -> dict[Symbol, List[Order]]:
        """
        Takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        
        result = {}
        
        global mid_price
        global price_history_banana
        global open_p
        global high_p
        global low_p
        global close_p
        
        for product in state.order_depths.keys():
            if product == 'PEARLS':
                bid_spread = 1
                ask_spread = 1
                open_bid_spread = 3
                open_ask_spread = 3
                start_trading = 0
                position_limit = 20
                position_spread = 15
                current_position = state.position.get(product,0)
                
                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []
                
                if state.timestamp >= start_trading:
                    if len(order_depth.sell_orders) > 0:
                        best_ask = min(order_depth.sell_orders.keys())
                        
                        if best_ask <= 10000-ask_spread:
                            best_ask_volume = order_depth.sell_orders[best_ask]
                        else:
                            best_ask_volume = 0
                    else:
                        best_ask_volume = 0
                         
                    if len(order_depth.buy_orders) > 0:
                        best_bid = max(order_depth.buy_orders.keys())
                    
                        if best_bid >= 10000+bid_spread:
                            best_bid_volume = order_depth.buy_orders[best_bid]
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
                        
                    if best_ask == 10000-open_ask_spread and -best_ask_volume > 0:
                        print("BUY", product, str(-best_ask_volume-open_ask_volume) + "x", 10000-open_ask_spread)
                        orders.append(Order(product, 10000-open_ask_spread, -best_ask_volume-open_ask_volume))
                    else:
                        if -best_ask_volume > 0:
                            print("BUY", product, str(-best_ask_volume) + "x", best_ask)
                            orders.append(Order(product, best_ask, -best_ask_volume))
                        if -open_ask_volume > 0:
                            print("BUY", product, str(-open_ask_volume) + "x", 10000-open_ask_spread)
                            orders.append(Order(product, 10000-open_ask_spread, -open_ask_volume))
                        
                    if best_bid == 10000+open_bid_spread and best_bid_volume > 0:
                        print("SELL", product, str(best_bid_volume+open_bid_volume) + "x", 10000+open_bid_spread)
                        orders.append(Order(product, 10000+open_bid_spread, -best_bid_volume-open_bid_volume))
                    else:
                        if best_bid_volume > 0:
                            print("SELL", product, str(best_bid_volume) + "x", best_bid)
                            orders.append(Order(product, best_bid, -best_bid_volume))
                        if open_bid_volume > 0:
                            print("SELL", product, str(open_bid_volume) + "x", 10000+open_bid_spread)
                            orders.append(Order(product, 10000+open_bid_spread, -open_bid_volume))
                        
                result[product] = orders
                
                
                
            if product == 'BANANAS':
                enough_data = True
                start_trading = 2000
                position_limit = 20
                position_spread = 15
                current_position = state.position.get(product, 0)
                history_length = 20
                
                
                
                """
                spread = 5
                spread_rate = 0.1

                buySpread = spread / 2
                sellSpread = spread / 2

                if (current_position) < 0:
                    buySpread = spread / 2 + current_position * spread_rate
                    sellSpread = spread - buySpread
                else:
                    sellSpread = spread / 2 + current_position * spread_rate
                    buySpread = spread - sellSpread
                """
                order_depth: OrderDepth = state.order_depths[product]
                orders: list[Order] = []
                
                price = 0
                count = 0
                for Trade in state.market_trades.get(product, []):
                    price += Trade.price * Trade.quantity
                    count += Trade.quantity

                if count == 0:
                    if len(price_history_banana) == 0:
                        enough_data = False
                    else:
                        current_avg_market_price = price_history_banana[-1]
                else:
                    current_avg_market_price = price / count
                
                if len(price_history_banana) > history_length:
                    price_history_banana = price_history_banana[1:]
                
        
                if state.timestamp >= start_trading and enough_data == True:
                    price_history_banana = np.append(price_history_banana, current_avg_market_price)
                    model = ARIMA(4,0,1)
                    pred = model.fit_predict(price_history_banana)
                    forecasted_price = model.forecast(pred, 1)[-1]
                        
                    open_p = np.append(open_p, price_history_banana[0])
                    high_p = np.append(high_p, max(order_depth.buy_orders.keys()))
                    low_p = np.append(low_p, min(order_depth.sell_orders.keys()))
                    close_p = np.append(close_p, price_history_banana[-1])
                     
                        
                    if len(open_p) > history_length:
                        open_p = open_p[1:]
                        high_p = high_p[1:]
                        low_p = low_p[1:]
                        close_p = close_p[1:]

                    spread = edge(open_p, high_p, low_p, close_p) * forecasted_price / 2
                    print("SPREAD", spread)
                    
                    sellSpread = spread
                    buySpread = spread
                    
                    
                    if len(order_depth.sell_orders) > 0:
                        best_ask = min(order_depth.sell_orders.keys())
                        
                        if best_ask <= forecasted_price - buySpread:
                            best_ask_volume = order_depth.sell_orders[best_ask]
                        else:
                            best_ask_volume = 0
                    else:
                        best_ask_volume = 0
                         
                    if len(order_depth.buy_orders) > 0:
                        best_bid = max(order_depth.buy_orders.keys())
                    
                        if best_bid >= forecasted_price + sellSpread:
                            best_bid_volume = order_depth.buy_orders[best_bid]
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

                    open_buy_spread = buySpread
                    open_sell_spread = sellSpread
                        
                    if best_ask == round(forecasted_price-open_buy_spread) and -best_ask_volume > 0:
                        print("BUY", product, str(-best_ask_volume-open_ask_volume) + "x", round(forecasted_price-open_buy_spread))
                        orders.append(Order(product, round(forecasted_price-open_buy_spread), -best_ask_volume-open_ask_volume))
                    else:
                        if -best_ask_volume > 0:
                            print("BUY", product, str(-best_ask_volume) + "x", best_ask)
                            orders.append(Order(product, best_ask, -best_ask_volume))
                        if -open_ask_volume > 0:
                            print("BUY", product, str(-open_ask_volume) + "x", round(forecasted_price-open_buy_spread))
                            orders.append(Order(product, round(forecasted_price-open_buy_spread), -open_ask_volume))
                        
                    if best_bid == round(forecasted_price+open_sell_spread) and best_bid_volume > 0:
                        print("SELL", product, str(best_bid_volume+open_bid_volume) + "x", round(forecasted_price+open_sell_spread))
                        orders.append(Order(product, round(forecasted_price+open_sell_spread), -best_bid_volume-open_bid_volume))
                    else:
                        if best_bid_volume > 0:
                            print("SELL", product, str(best_bid_volume) + "x", best_bid)
                            orders.append(Order(product, best_bid, -best_bid_volume))
                        if open_bid_volume > 0:
                            print("SELL", product, str(open_bid_volume) + "x", round(forecasted_price+open_sell_spread))
                            orders.append(Order(product, round(forecasted_price+open_sell_spread), -open_bid_volume))

                result[product] = orders
        logger.flush(state, orders)
        return result

