#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List


def get_sma(prices, rate):
    #return prices.rolling(rate).mean()
    return np.average(prices)

def get_std(prices, rate):
    #return prices.rolling(rate).std()
    return np.std(prices)

price_history_banana = np.array([])
price_history_pearls = np.array([])

"""
Executes the trades
"""
class Trader:
    def __init__(self):
        self.holdings = 0
        self.last_trade = 0

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        
        result = {}
        
        global price_history_banana
        global price_history_pearls
        
        for product in state.order_depths.keys():
            if product == 'BANANAS':
                    
                start_trading = 2100
                position_limit = 20
                current_position = state.position.get(product,0)
                history_length = 10
                spread = 3
                
                order_depth: OrderDepth = state.order_depths[product]

                price = 0
                count = 0.000001

                for Trade in state.market_trades.get(product, []):
                    price += Trade.price * Trade.quantity
                    count += Trade.quantity
                current_avg_market_price = price / count
                
                prev_sma = np.average(price_history_pearls)
                
                price_history_pearls = np.append(price_history_pearls, current_avg_market_price)
                if len(price_history_pearls) >= history_length+1:
                    price_history_pearls = price_history_pearls[1:]
                
                current_sma = np.average(price_history_pearls)
                std = np.std(price_history_pearls)
                
                orders: list[Order] = []
                
                rate = 20
                m = 2 # of std devs
                    
                if state.timestamp >= start_trading:

                    #df_pearl_prices = pd.DataFrame(price_history_pearls, columns=['mid_price'])
                    
                    #sma = get_sma(df_pearl_prices['mid_price'], rate).to_numpy()
                    #std = get_std(df_pearl_prices['mid_price'], rate).to_numpy()

                    #upper_curr = sma[-1] + m * std
                    #upper_prev = sma[-2] + m * std
                    #lower_curr = sma[-1] - m * std
                    #lower_prev = sma[-2] - m * std
                    upper_curr = current_sma + m * std
                    upper_prev = prev_sma + m * std
                    lower_curr = current_sma - m * std
                    lower_prev = prev_sma - m * std

                    if len(order_depth.sell_orders) > 0:

                        best_ask = min(order_depth.sell_orders.keys())
                        best_ask_volume = order_depth.sell_orders[best_ask]

                        if price_history_pearls[-2] > lower_prev and best_ask <= lower_curr and np.abs(best_ask_volume) > 0:
                            print("BUY", product, str(-best_ask_volume) + "x", best_ask)
                            orders.append(Order(product, best_ask, -best_ask_volume))

                    if len(order_depth.buy_orders) != 0:
                        best_bid = max(order_depth.buy_orders.keys())
                        best_bid_volume = order_depth.buy_orders[best_bid]
                       
                        if price_history_pearls[-2] < upper_prev and best_bid >= upper_curr and best_bid_volume > 0:
                            print("SELL", product, str(best_bid_volume) + "x", best_bid)
                            orders.append(Order(product, best_bid, -best_bid_volume))

                        
                        
                result[product] = orders
                
                
                
            # if product == 'BANANAS':
                
            #     start_trading = 2000
            #     position_limit = 20
            #     current_position = state.position.get(product,0)
            #     history_length = 20
            #     spread = 3
                
            #     order_depth: OrderDepth = state.order_depths[product]

            #     price = 0
            #     count = 0.000001

            #     for Trade in state.market_trades.get(product, []):
            #         price += Trade.price * Trade.quantity
            #         count += Trade.quantity
            #     current_avg_market_price = price / count
                
            #     price_history_banana = np.append(price_history_banana, current_avg_market_price)
            #     if len(price_history_banana) == history_length+1:
            #         price_history_banana = price_history_banana[1:]
                
            #     orders: list[Order] = []

            #     if state.timestamp >= start_trading:

            #         df_banana_prices = pd.DataFrame(price_history_banana, columns=['mid_price'])
                    
            #         sma = get_sma(df_banana_prices, rate)[-1]
            #         std = get_std(df_banana_prices, rate)[-1]

            #         upper = sma + m * std
            #         lower = sma - m * std

            #         if len(order_depth.sell_orders) > 0:
            #             best_ask = min(order_depth.sell_orders.keys())
                        
            #             if best_ask <= lower:
            #                 best_ask_volume = order_depth.sell_orders[best_ask]
            #             else:
            #                 best_ask_volume = 0
            #         else:
            #             best_ask_volume = 0
                         
            #         if len(order_depth.buy_orders) > 0:
            #             best_bid = max(order_depth.buy_orders.keys())
                    
            #             if best_bid >= upper:
            #                 best_bid_volume = order_depth.buy_orders[best_bid]
            #             else:
            #                 best_bid_volume = 0 
            #         else:
            #             best_bid_volume = 0
                
            #         if current_position - best_ask_volume - best_bid_volume > position_limit:
            #             best_ask_volume = current_position - best_bid_volume - position_limit
            #         elif current_position - best_ask_volume - best_bid_volume < -position_limit:
            #             best_bid_volume = current_position + position_limit - best_ask_volume
                        
            #         position_spread = 15
            #         open_ask_volume = current_position - position_spread - best_ask_volume
            #         open_bid_volume = current_position + position_spread - best_bid_volume
                        
            #         if best_ask == lower and -best_ask_volume > 0:
            #             print("BUY", product, str(-best_ask_volume-open_ask_volume) + "x", lower)
            #             orders.append(Order(product, lower, -best_ask_volume-open_ask_volume))
            #         else:
            #             if -best_ask_volume > 0:
            #                 print("BUY", product, str(-best_ask_volume) + "x", best_ask)
            #                 orders.append(Order(product, best_ask, -best_ask_volume))
            #             if -open_ask_volume > 0:
            #                 print("BUY", product, str(-open_ask_volume) + "x", lower)
            #                 orders.append(Order(product, lower, -open_ask_volume))
                        
            #         if best_bid == upper and best_bid_volume > 0:
            #             print("SELL", product, str(best_bid_volume+open_bid_volume) + "x", upper)
            #             orders.append(Order(product, upper, best_bid_volume+open_bid_volume))
            #         else:
            #             if best_bid_volume > 0:
            #                 print("SELL", product, str(best_bid_volume) + "x", best_bid)
            #                 orders.append(Order(product, best_bid, -best_bid_volume))
            #             if open_bid_volume > 0:
            #                 print("SELL", product, str(open_bid_volume) + "x", upper)
            #                 orders.append(Order(product, lower, -open_bid_volume))
            #     result[product] = orders
                
            """
            Profit Calculator
            """     
            timestamp = self.last_trade;
            for trades in state.own_trades.values():
                for trade in trades:
                    if trade.timestamp != self.last_trade:
                        timestamp = trade.timestamp
                        if trade.buyer == "SUBMISSION":
                            self.holdings += trade.price * trade.quantity
                        else:
                            self.holdings -= trade.price * trade.quantity
            self.last_trade = timestamp
            profit = 0
            for product in state.position:
                profit += state.position[product] * (max(state.order_depths[product].buy_orders) + min(state.order_depths[product].sell_orders)) / 2
            profit -= self.holdings
            #print("Profit: " + str(profit))

        return result


# In[5]:





# In[7]:





# In[ ]:




