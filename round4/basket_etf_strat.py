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

##################################################################################

class Trader:
    baguette_limit = 150
    dip_limit = 300
    ukulele_limit = 70
    basket_limit = 70
    basket_pnav_ratio = 1.0051
    basket_eps = 0.002
    basket_eps_open = basket_eps * 2
    etf_returns = np.array([])
    asset_returns = np.array([])

    def __init__(self):
        self.basket_prev: float = None
        self.baguette_prev: float = None
        self.dip_prev: float = None
        self.ukulele_prev: float = None

    def run(self, state: TradingState) -> dict[Symbol, List[Order]]:
        result = {}
        
        for product in state.order_depths.keys():
            orders_picnic_basket: list[Order] = []
            orders_baguette: list[Order] = []
            orders_dip: list[Order] = []
            orders_ukulele: list[Order] = []

            if product == 'PICNIC_BASKET':
                # current positions
                basket_pos = state.position.get("PICNIC_BASKET", 0)
                baguette_pos = state.position.get("BAGUETTE", 0)
                dip_pos = state.position.get("DIP", 0)
                ukulele_pos = state.position.get("UKULELE", 0)


                basket_buy_orders: List[Order] = state.order_depths[product].buy_orders
                basket_sell_orders: List[Order] = state.order_depths[product].sell_orders

                basket_best_bid: float = max(basket_buy_orders)
                basket_best_ask: float = min(basket_sell_orders)

                # Finding price / NAV ratio
                basket_price: float = (basket_best_bid + basket_best_ask) / 2

                baguette_buy_orders: List[Order] = state.order_depths['BAGUETTE'].buy_orders
                baguette_sell_orders: List[Order] = state.order_depths['BAGUETTE'].sell_orders

                baguette_best_bid: float = max(baguette_buy_orders)
                baguette_best_ask: float = min(baguette_sell_orders)

                baguette_price: float = (baguette_best_bid + baguette_best_ask) / 2
 
                dip_buy_orders: List[Order] = state.order_depths['DIP'].buy_orders
                dip_sell_orders: List[Order] = state.order_depths['DIP'].sell_orders

                dip_best_bid: float = max(dip_buy_orders)
                dip_best_ask: float = min(dip_sell_orders)

                dip_price: float = (dip_best_bid + dip_best_ask) / 2

                ukulele_buy_orders: List[Order] = state.order_depths['UKULELE'].buy_orders
                ukulele_sell_orders: List[Order] = state.order_depths['UKULELE'].sell_orders

                ukulele_best_bid: float = max(ukulele_buy_orders)
                ukulele_best_ask: float = min(ukulele_sell_orders)

                ukulele_price: float = (ukulele_best_bid + ukulele_best_ask) / 2

                est_price: float = 4 * dip_price + 2 * baguette_price + ukulele_price

                price_nav_ratio: float = basket_price / est_price
                print("HEY", price_nav_ratio, self.basket_pnav_ratio)

                
                
                self.etf_returns = np.append(self.etf_returns, basket_price)
                self.asset_returns = np.append(self.asset_returns, est_price)

                rolling_mean_etf = np.mean(self.etf_returns[-10:])
                rolling_std_etf = np.std(self.etf_returns[-10:])

                rolling_mean_asset = np.mean(self.asset_returns[-10:])
                rolling_std_asset = np.std(self.asset_returns[-10:])

                z_score_etf = (self.etf_returns[-1] - rolling_mean_etf) / rolling_std_etf
                z_score_asset = (self.asset_returns[-1] - rolling_mean_asset) / rolling_std_asset

                z_score_diff = z_score_etf - z_score_asset

                #if z_score_diff > 2:
                    # Buy the underlying asset and short sell the ETF
                #elif z_score_diff < -2:
                    # Sell the underlying asset and buy the ETF
               
            
                
                #if price_nav_ratio < self.basket_pnav_ratio - self.basket_eps:
                if z_score_diff < -2:
                    # ETF is undervalued! -> we buy ETF and sell individual assets!
                    # Finds volume to buy that is within position limit
                    #basket_best_ask_vol = max(basket_pos-self.basket_limit, state.order_depths['PICNIC_BASKET'].sell_orders[basket_best_ask])
                    basket_best_ask_vol = state.order_depths['PICNIC_BASKET'].sell_orders[basket_best_ask]
                    print("BUY", 'PICNIC_BASKET', -basket_best_ask_vol, "x", basket_best_ask)
                    orders_picnic_basket.append(Order('PICNIC_BASKET', basket_best_ask, -basket_best_ask_vol))
                    
                    #baguette_best_bid_vol = min(self.baguette_limit-baguette_pos, state.order_depths['BAGUETTE'].buy_orders[baguette_best_bid])
                    baguette_best_bid_vol =  state.order_depths['BAGUETTE'].buy_orders[baguette_best_bid]
                    print("SELL", "BAGUETTE", baguette_best_bid_vol, "x", baguette_best_bid)
                    orders_baguette.append(Order("BAGUETTE", baguette_best_bid, -baguette_best_bid_vol))
                    
                    #dip_best_bid_vol = min(self.dip_limit-dip_pos, state.order_depths['DIP'].buy_orders[dip_best_bid])
                    dip_best_bid_vol = state.order_depths['DIP'].buy_orders[dip_best_bid]
                    print("SELL", "DIP", dip_best_bid_vol, "x", dip_best_bid)
                    orders_dip.append(Order("DIP", dip_best_bid, -dip_best_bid_vol))
                    
                    #ukulele_best_bid_vol = min(self.ukulele_limit-ukulele_pos, state.order_depths['UKULELE'].buy_orders[ukulele_best_bid])
                    ukulele_best_bid_vol = state.order_depths['UKULELE'].buy_orders[ukulele_best_bid]
                    print("SELL", "UKULELE", ukulele_best_bid_vol, "x", ukulele_best_bid)
                    orders_ukulele.append(Order("UKULELE", ukulele_best_bid, -ukulele_best_bid_vol))
                    
                    
                #elif price_nav_ratio > self.basket_pnav_ratio + self.basket_eps:
                elif z_score_diff > 2:
                    # ETF is overvalued! -> we sell ETF and buy individual assets!
                    # Finds volume to buy that is within position limit
                    #basket_best_bid_vol = min(self.basket_limit-basket_pos, state.order_depths['PICNIC_BASKET'].buy_orders[basket_best_bid])
                    basket_best_bid_vol = state.order_depths['PICNIC_BASKET'].buy_orders[basket_best_bid]
                    print("SELL", 'PICNIC_BASKET', basket_best_bid_vol, "x", basket_best_bid)
                    orders_picnic_basket.append(Order('PICNIC_BASKET', basket_best_bid, -basket_best_bid_vol))
                    
                    #baguette_best_ask_vol = max(baguette_pos-self.baguette_limit, state.order_depths['BAGUETTE'].sell_orders[baguette_best_ask])
                    baguette_best_ask_vol = state.order_depths['BAGUETTE'].sell_orders[baguette_best_ask]
                    print("BUY", "BAGUETTE", -baguette_best_ask_vol, "x", baguette_best_ask)
                    orders_baguette.append(Order("BAGUETTE", baguette_best_ask, -baguette_best_ask_vol))
                    
                    #dip_best_ask_vol = max(dip_pos-self.dip_limit, state.order_depths['DIP'].sell_orders[dip_best_ask])
                    dip_best_ask_vol = state.order_depths['DIP'].sell_orders[dip_best_ask]
                    print("BUY", "DIP", -dip_best_ask_vol, "x", dip_best_ask)
                    orders_dip.append(Order("DIP", dip_best_ask, -dip_best_ask_vol))
                    
                    #ukulele_best_ask_vol = max(ukulele_pos-self.ukulele_limit, state.order_depths['UKULELE'].sell_orders[ukulele_best_ask])
                    ukulele_best_ask_vol = state.order_depths['UKULELE'].sell_orders[ukulele_best_ask]
                    print("BUY", "UKULELE", -ukulele_best_ask_vol, "x", ukulele_best_ask)
                    orders_ukulele.append(Order("UKULELE", ukulele_best_ask, -ukulele_best_ask_vol))

                result['PICNIC_BASKET'] = orders_picnic_basket
                result['BAGUETTE'] = orders_baguette
                result['DIP'] = orders_dip
                result['UKULELE'] = orders_ukulele
        
        logger.flush(state, orders_picnic_basket+orders_baguette+orders_dip+orders_ukulele) # type: ignore
        return result