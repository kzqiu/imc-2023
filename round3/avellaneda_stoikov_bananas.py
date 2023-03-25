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

# Avellaneda-Stoikov reservation price
def reservation_price(mid_price: float, position: int, gamma: float, var: float, timestamp: int, time_horizon: float) -> float:
    return mid_price - position * gamma * var * (1 - timestamp / time_horizon)

# Avellaneda-Stoikov bid-ask spread (1/2 if symmetrical!)
def bid_ask_spread(gamma: float, var: float, kappa: float, timestamp: int, time_horizon: float):
    return gamma * var * (1 - timestamp / time_horizon) + 2 / gamma * np.log(1 + gamma / kappa)

def get_variance():
    return np.exp(np.random.normal(loc=0.2013577184710968, scale=0.6244283309726839))

##################################################################################

class Trader:
    def __init__(self):
        self.gamma = 0.1 # try 0.5?
        self.kappa = 2 # market liquidity
        self.time_horizon = 100000.0
        self.A = 292.991854896

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

                mid = mid_price(best_bid, best_ask)
                var = get_variance()

                # Avellaneda-Stoikov Reservation price estimate
                fair_price = reservation_price(mid, current_position, 
                                               self.gamma, var, state.timestamp, self.time_horizon)
                
                # Getting spread!
                full_spread = bid_ask_spread(self.gamma, var, self.kappa, state.timestamp, self.time_horizon)

                """
                optimal prices:
                fair_price +/- fullspread
                """
                reserve_ask = fair_price + full_spread / 2
                reserve_bid = fair_price - full_spread / 2

                delta_ask = reserve_ask - mid
                delta_bid = mid - reserve_bid

                lambda_ask = self.A * np.exp(-self.kappa * delta_ask)
                lambda_bid = self.A * np.exp(-self.kappa * delta_bid)

                prob_a = lambda_ask * 100 / self.time_horizon
                prob_b = lambda_bid * 100 / self.time_horizon

                print(f'price: {fair_price}, spread: {full_spread}')
                print(f'lambdas: bid = {lambda_bid}, ask = {lambda_ask}')
                print(f'delta: bid = {delta_bid}, ask = {delta_ask}')
                print(f'probs: bid = {prob_b}, ask = {prob_a}')

                ask_arrival_rnd = np.random.rand()
                bid_arrival_rnd = np.random.rand()

                # Trading code
                sells = sorted([it for it in sell_orders.items()], reverse=False)
                buys = sorted([it for it in buy_orders.items()], reverse=True)

                agg_vol = current_position

                # trading
                if best_bid >= fair_price - delta_ask and agg_vol != -20 and prob_a > ask_arrival_rnd:
                    # print("SELL", product, 1, "x", best_bid)
                    # orders.append(Order(product, best_bid, -1))

                    for order in buys:
                        bid = order[0]
                        vol = order[1]

                        real_vol = max(-position_limit - agg_vol, -vol)

                        if bid >= fair_price - delta_bid:
                            print("SELL", product, str(real_vol) + "x", bid)
                            agg_vol += real_vol
                            orders.append(Order(product, bid, real_vol))

                agg_vol = current_position

                if best_ask <= fair_price + delta_ask and agg_vol != 20 and prob_b > bid_arrival_rnd:
                    # print("BUY", product, 1, "x", best_ask)
                    # orders.append(Order(product, best_ask, 1))

                    for order in sells:
                        ask = order[0]
                        vol = order[1]

                        real_vol = min(position_limit - agg_vol, -vol)

                        if ask <= fair_price + delta_ask:
                            print("BUY", product, str(real_vol) + "x", ask)
                            agg_vol += real_vol 
                            orders.append(Order(product, ask, real_vol))


                result[product] = orders
        
        logger.flush(state, orders) # type: ignore
        return result