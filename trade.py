from Param import Param
import numpy as np
import math
import copy

import dataHelper as dh

class Signal:
    LONG = 1
    SHORT = 0
    HOLD = -1

    current_pos = HOLD

def trade(predictions, file_number):
    # Array for accumulating signals until we have enough information to act
    x = np.full([predictions.shape[0], Param.n_out], np.inf)

    actual = dh.get_testing_output(Param.filenames[file_number])
    actual = dh.get_column(Param.col_to_trade_on, actual)

    # will hold the action that we took on each tick
    d = np.full([predictions.shape[0], 1], np.inf)

    profit = 0
    gains = 0
    losses = 0
    prev_profit = 0

    for i in range(predictions.shape[0] - Param.n_out - 1):
        # have we made any money this time step?
        prev_profit = copy.deepcopy(profit)
        profit = check_position(profit, actual[i, 0] - actual[i-1, 0])
        if profit - prev_profit > 0:
            gains += profit - prev_profit
        else:
            losses += prev_profit - profit
        # END if

        curr = np.reshape(actual[i, 0], (-1,1))
        future = predictions[i+1, :]

        window = np.concatenate((curr, future))

        for j in range(1, Param.n_out + 1):
            if window[j] - window[j-1] > 0:
                is_full = add_signal(x, i+j, Signal.LONG)
            elif window[j] - window[j-1] < 0:
                is_full = add_signal(x, i+j, Signal.SHORT)
            else:
                pass
            # END if

            if is_full:
                d[i+j-1] = make_trade(x, i+j-1, actual[i+j-1, 0], predictions[i+j, 0])
                Signal.current_pos = d[i+j-1]
            # END if
        # END for
    # END for
    print("Gains/Losses for stock " + str(file_number)+ ": " + str(gains/losses))
    return profit
            
def add_signal(x, t, signal):
    is_full = False
    for i in range(Param.n_out):
        if np.isinf(x[t, i]):
            x[t, i] = signal
            if i == Param.n_out - 1:
                is_full = True
            # END if
            break
        # END if
    # END for
    return is_full
# END add_signal

def make_trade(x, t, recent_actual, predicted):
    total_signal = np.sum(x[t,:])
    if np.isinf(total_signal):
        return np.inf
    elif total_signal >= 2:
        signal = Signal.LONG
    else:
        signal = Signal.SHORT
    # END if 

    predicted = predicted[0]
    recent_actual = recent_actual[0]

    if np.abs((predicted - recent_actual)/recent_actual) > 0.001:
        return signal
    else:
        return Signal.HOLD
    # END if
# END make_trade

def check_position(profit, delta):
    mult = 1
    if Signal.current_pos == Signal.SHORT:
        mult = -1

    if Signal.current_pos == Signal.SHORT or Signal.current_pos == Signal.LONG:
        profit += delta*mult

    return profit