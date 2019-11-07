import math
import numpy as np

from Param import Param

def sma(period, close_prices):
    sma = np.zeros([close_prices.shape[0] - period, 1])

    for i in range(period, close_prices.shape[0]):
        sma[i - period] = sum(close_prices.iloc[range(i-period), i])/period
    # END for
    return sma
#END sma

def roc(period, close_prices):
    roc = np.zeros([close_prices.shape[0] - period, 1])
    roc[range(period)] = close_prices.iloc[range(period)]

    for i in range(period, close_prices.shape[0]):
        roc[i - period] = 100*(close_prices[i] - close_prices[i - period])/close_prices[i-period]
    #END for

    return roc
# END roc

def rsi(period, close_prices):
    rsi = np.zeros([close_prices.shape[0] - period, 1])
    for i in range(period, len(close_prices)):
        
        rs = rs(close_prices[range(i-period, period + 1)])
        rsi[i - period] = 100 - (100/(1+rs))
    #END for
#END rsi

def rs(prices):
    total_gain = 0
    total_loss = 0
    num_gain = 0
    num_loss = 0

    for i in range(1, len(prices)):
        close_diff = (prices[i] - prices[i-1])/prices[i-1]
        if close_diff > 0:
            total_gain += close_diff
            num_gain += 1
        else:
            total_loss += close_diff
            num_loss += 1
        #END if
    #END for
    return (total_gain/num_gain)/(total_loss/num_loss)
#END rs

def typical_price(high_prices, low_prices, close_prices):
    tp = np.add(high_prices, low_prices)
    tp = np.add(tp, close_prices)/3
    return tp
#END tp

def cci(high_prices, low_prices, close_prices, period):
    tp = np.zeros(high_prices.shape)
    sma_tp = np.zeros(high_prices.shape[0] - period, 1)
    cci = np.zeros(high_prices.shape[0] - period, 1)
    tp = typical_price(high_prices, low_prices, close_prices)
    
    sma_tp = sma(period, tp)
    tp = tp[range(period, len(high_prices))]

    for i in range(period, len(high_prices)):
        md = np.std(tp[range(i - period, period + 1)])
        cci[i - period] = (tp[i] - sma_tp[i - period - 1])/(0.015*md)
    #END for

    return cci
#END cci

def sto(high_prices, low_prices, close_prices, period):
    sto = np.zeros(high_prices.shape[0] - period, 1)

    for i in range(period, len(sto)):
        highest_high = np.max(high_prices[range(i-period, i)])
        lowest_low = np.min(low_prices[range(i-period, i)])

        sto[i - period] = 100*(close_prices[i] - lowest_low)/(highest_high - lowest_low)
    #END for

    return sto
#END sto

def atr(high_prices, low_prices, close_prices, period):
    atr = np.zeros(high_prices.shape[0] - period - 1, 1)
    tr = true_range(high_prices, low_prices, close_prices)

    for i in range(period, len(tr)):
        atr[i - period] = np.sum(tr[range(i - period, i)])
    #END for

    return atr
# END atr

def true_range(high_prices, low_prices, close_prices):
    tr = np.zeros(high_prices.shape[0] - 1, 1)
    for i in range(len(high_prices)):
        tr[i] = max([high_prices[i] - low_prices[i], np.abs(high_prices[i] - close_prices[i-1]), np.abs(low_prices[i] - close_prices[i-1])])
    #END for

    return tr
# END true_range