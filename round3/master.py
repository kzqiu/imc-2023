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

def get_zscore(S1, S2, window1, window2):
    
    # If window length is 0, algorithm doesn't make sense, so exit
    if (window1 == 0) or (window2 == 0):
        return 0

    # Compute rolling mean and rolling standard deviation
    ratios = np.divide(np.asarray(S1),np.asarray(S2))
    
    ma1 = np.mean(ratios[-window1:])
    ma2 = np.mean(ratios[-window2:])
    std = np.std(ratios[-window1:])
    
    zscore = (ma1 - ma2)/std

    return [zscore, ratios[-1]]

class Trader:
    def __init__(self):
        self.coconuts_data = [8000.0 for i in range(100)]
        self.pina_coladas_data = [15000.0 for i in range(100)]
        self.dolphin_data = [3074.0 for i in range(100)]
        self.gear_data = [99100.0 for i in range(100)]
        self.basket_prev = None
        self.baguette_prev = None
        self.dip_prev = None
        self.ukulele_prev = None
        self.etf_returns = np.array([])
        self.asset_returns = np.array([])


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
        orders_picnic_basket: list[Order] = []
        orders_baguette: list[Order] = []
        orders_dip: list[Order] = []
        orders_ukulele: list[Order] = []

        # Iterate over all the keys (the available products) contained in the order depths
        for product in state.order_depths.keys():
            
            if product == 'PEARLS':
                spread = 1
                open_spread = 3
                position_limit = 20
                position_spread = 15
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
                window1 = 120
                window2 = 5
                
                order_depth_coconuts: OrderDepth = state.order_depths['COCONUTS']
                order_depth_pina_coladas: OrderDepth = state.order_depths['PINA_COLADAS']

                mid_price_coconuts = (min(order_depth_coconuts.sell_orders.keys()) + max(order_depth_coconuts.buy_orders.keys()))/2
                mid_price_pina_coladas = (min(order_depth_pina_coladas.sell_orders.keys()) + max(order_depth_pina_coladas.buy_orders.keys()))/2

                self.coconuts_data.append(mid_price_coconuts)
                self.pina_coladas_data.append(mid_price_pina_coladas)
                

                couple = get_zscore(self.pina_coladas_data, self.coconuts_data, window1, window2)
                zscore = couple[0]
                ratio = couple[1]
                coconuts_position = state.position.get('COCONUTS', 0)
                pina_coladas_position = state.position.get('PINA_COLADAS', 0)


                if zscore > 1 and pina_coladas_position < 300 and coconuts_position > -600:
                    if len(order_depth_pina_coladas.buy_orders.keys()) > 0:
                        best_bid = max(order_depth_pina_coladas.buy_orders.keys())
                        best_bid_volume = order_depth_pina_coladas.buy_orders[best_bid]
                        print("SELL PINA_COLADAS", str(best_bid_volume) + "x", best_bid)
                        orders_pina_coladas.append(Order('PINA_COLADAS', best_bid, -best_bid_volume))
                    #sell coconuts
                    if len(order_depth_coconuts.buy_orders.keys()) > 0:
                        best_bid = max(order_depth_coconuts.buy_orders.keys())
                        best_bid_volume = order_depth_coconuts.buy_orders[best_bid]
                        print("SELL COCONUTS", str(best_bid_volume) + "x", best_bid)
                        orders_coconuts.append(Order('COCONUTS', best_bid, -best_bid_volume*ratio))

                elif zscore < -1 and pina_coladas_position > -300 and coconuts_position < 600:
                    #short position
                    #buy coconuts
                    if len(order_depth_coconuts.sell_orders.keys()) > 0:
                        best_ask = min(order_depth_coconuts.sell_orders.keys())
                        best_ask_volume = order_depth_coconuts.sell_orders[best_ask]
                        print("BUY COCONUTS", str(-best_ask_volume) + "x", best_ask)
                        orders_coconuts.append(Order('COCONUTS', best_ask, -best_ask_volume*ratio))
                    #sell pina_coladas
                    if len(order_depth_pina_coladas.sell_orders.keys()) > 0:
                        best_ask = min(order_depth_pina_coladas.sell_orders.keys())
                        best_ask_volume = order_depth_pina_coladas.sell_orders[best_ask]
                        print("BUY PINA_COLADAS", str(-best_ask_volume) + "x", best_ask)
                        orders_pina_coladas.append(Order('PINA_COLADAS', best_ask, -best_ask_volume))
                    
                elif abs(zscore) < 0.5 or (abs(pina_coladas_position) >= 300 and abs(coconuts_position) >= 600):
                # elif abs(zscore) < 0.5:
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

##################################################################################

            if product == "DIVING_GEAR":
                window1 = 5
                window2 = 20
                
                order_depth_diving_gear: OrderDepth = state.order_depths['DIVING_GEAR']

                mid_price_diving_gear = (min(order_depth_diving_gear.sell_orders.keys()) + max(order_depth_diving_gear.buy_orders.keys()))/2

                self.dolphin_data.append(dolphins)
                self.gear_data.append(mid_price_diving_gear)
                

                couple = get_zscore(self.gear_data, self.dolphin_data, window1, window2)
                zscore = -couple[0]
                ratio = couple[1]
                diving_gear_position = state.position.get('DIVING_GEAR', 0)


                if zscore > 1 and diving_gear_position < 50:
                    if len(order_depth_diving_gear.buy_orders.keys()) > 0:
                        best_bid = max(order_depth_diving_gear.buy_orders.keys())
                        best_bid_volume = order_depth_diving_gear.buy_orders[best_bid]
                        print("SELL DIVING_GEAR", str(best_bid_volume) + "x", best_bid)
                        orders_diving_gear.append(Order('DIVING_GEAR', best_bid, -best_bid_volume))

                elif zscore < -1 and diving_gear_position > -50:
                    #short position
                    #sell pina_coladas
                    if len(order_depth_diving_gear.sell_orders.keys()) > 0:
                        best_ask = min(order_depth_diving_gear.sell_orders.keys())
                        best_ask_volume = order_depth_diving_gear.sell_orders[best_ask]
                        print("BUY DIVING_GEAR", str(-best_ask_volume) + "x", best_ask)
                        orders_diving_gear.append(Order('DIVING_GEAR', best_ask, -best_ask_volume))
                    
                # elif abs(zscore) < 0.5 or (abs(pina_coladas_position) >= 300 and abs(coconuts_position) >= 600):
                elif abs(zscore) < 0.5 or abs(diving_gear_position) > 0:
                    # Reset Positions to 0

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
                    
            if product == 'PICNIC_BASKET':
                # current positions
                basket_pos = state.position.get("PICNIC_BASKET", 0)
                baguette_pos = state.position.get("BAGUETTE", 0)
                dip_pos = state.position.get("DIP", 0)
                ukulele_pos = state.position.get("UKULELE", 0)

##################################################################################
                
                basket_buy_orders: Dict[int, int] = state.order_depths[product].buy_orders
                basket_sell_orders: Dict[int, int] = state.order_depths[product].sell_orders

                basket_best_bid: float = max(basket_buy_orders)
                basket_best_ask: float = min(basket_sell_orders)

                # Finding price / NAV ratio
                basket_price: float = (basket_best_bid + basket_best_ask) / 2

                baguette_buy_orders: Dict[int, int] = state.order_depths['BAGUETTE'].buy_orders
                baguette_sell_orders: Dict[int, int] = state.order_depths['BAGUETTE'].sell_orders

                baguette_best_bid: float = max(baguette_buy_orders)
                baguette_best_ask: float = min(baguette_sell_orders)

                baguette_price: float = (baguette_best_bid + baguette_best_ask) / 2
 
                dip_buy_orders: Dict[int, int] = state.order_depths['DIP'].buy_orders
                dip_sell_orders: Dict[int, int] = state.order_depths['DIP'].sell_orders

                dip_best_bid: float = max(dip_buy_orders)
                dip_best_ask: float = min(dip_sell_orders)

                dip_price: float = (dip_best_bid + dip_best_ask) / 2

                ukulele_buy_orders: Dict[int, int] = state.order_depths['UKULELE'].buy_orders
                ukulele_sell_orders: Dict[int, int] = state.order_depths['UKULELE'].sell_orders

                ukulele_best_bid: float = max(ukulele_buy_orders)
                ukulele_best_ask: float = min(ukulele_sell_orders)

                ukulele_price: float = (ukulele_best_bid + ukulele_best_ask) / 2

                est_price: float = 4 * dip_price + 2 * baguette_price + ukulele_price

                price_nav_ratio: float = basket_price / est_price

##################################################################################

                self.etf_returns = np.append(self.etf_returns, basket_price)
                self.asset_returns = np.append(self.asset_returns, est_price)

                rolling_mean_etf = np.mean(self.etf_returns[-10:])
                rolling_std_etf = np.std(self.etf_returns[-10:])

                rolling_mean_asset = np.mean(self.asset_returns[-10:])
                rolling_std_asset = np.std(self.asset_returns[-10:])

                z_score_etf = (self.etf_returns[-1] - rolling_mean_etf) / rolling_std_etf
                z_score_asset = (self.asset_returns[-1] - rolling_mean_asset) / rolling_std_asset

                z_score_diff = z_score_etf - z_score_asset

                print(f'ZSCORE DIFF = {z_score_diff}')

                # implement stop loss
                # stop_loss = 0.01

                #if price_nav_ratio < self.basket_pnav_ratio - self.basket_eps:
                if z_score_diff < -2:
                    # stop_loss_price = self.etf_returns[-2] 


                    # ETF is undervalued! -> we buy ETF and sell individual assets!
                    # Finds volume to buy that is within position limit
                    #basket_best_ask_vol = max(basket_pos-self.basket_limit, state.order_depths['PICNIC_BASKET'].sell_orders[basket_best_ask])
                    basket_best_ask_vol = state.order_depths['PICNIC_BASKET'].sell_orders[basket_best_ask]
                    baguette_best_bid_vol =  state.order_depths['BAGUETTE'].buy_orders[baguette_best_bid]
                    dip_best_bid_vol = state.order_depths['DIP'].buy_orders[dip_best_bid]
                    ukulele_best_bid_vol = state.order_depths['UKULELE'].buy_orders[ukulele_best_bid]

                    limit_mult = min(-basket_best_ask_vol, ukulele_best_bid_vol, 
                                     round(baguette_best_bid_vol / 2), round(dip_best_bid_vol / 4))

                    print(f'LIMIT: {limit_mult}')

                    print("BUY", 'PICNIC_BASKET', limit_mult, "x", basket_best_ask)
                    orders_picnic_basket.append(Order('PICNIC_BASKET', basket_best_ask, limit_mult))
                    
                #elif price_nav_ratio > self.basket_pnav_ratio + self.basket_eps:
                elif z_score_diff > 2:
                    # ETF is overvalued! -> we sell ETF and buy individual assets!
                    # Finds volume to buy that is within position limit
                    #basket_best_bid_vol = min(self.basket_limit-basket_pos, state.order_depths['PICNIC_BASKET'].buy_orders[basket_best_bid])
                    basket_best_bid_vol = state.order_depths['PICNIC_BASKET'].buy_orders[basket_best_bid]
                    baguette_best_ask_vol = state.order_depths['BAGUETTE'].sell_orders[baguette_best_ask]
                    dip_best_ask_vol = state.order_depths['DIP'].sell_orders[dip_best_ask]
                    ukulele_best_ask_vol = state.order_depths['UKULELE'].sell_orders[ukulele_best_ask]

                    limit_mult = min(basket_best_bid_vol, -ukulele_best_ask_vol, 
                                     round(-baguette_best_ask_vol / 2), round(-dip_best_ask_vol / 4))

                    print(f'LIMIT: {limit_mult}')

                    print("SELL", 'PICNIC_BASKET', limit_mult, "x", basket_best_bid)
                    orders_picnic_basket.append(Order('PICNIC_BASKET', basket_best_bid, -limit_mult))

        # Add all the above orders to the result dict
        result['PEARLS'] = orders_pearls
        result['DIVING_GEAR'] = orders_diving_gear
        result['COCONUTS'] = orders_coconuts
        result['PINA_COLADAS'] = orders_pina_coladas
        result['BERRIES'] = orders_berries
        result['PICNIC_BASKET'] = orders_picnic_basket
        # result['BAGUETTE'] = orders_baguette
        # result['DIP'] = orders_dip
        # result['UKULELE'] = orders_ukulele
        # Return the dict of orders
        # These possibly contain buy or sell orders for PEARLS
        # Depending on the logic above
        logger.flush(state, orders_diving_gear + orders_coconuts 
                     + orders_pina_coladas + orders_berries + orders_pearls + orders_picnic_basket + orders_baguette _ orders_dip + orders_ukulele)

        return result