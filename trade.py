from Param import Param
import numpy as np
import math

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

    # will hold the action that we took on each tick
    d = np.full([predictions.shape[0], 1], np.inf)

    profit = 0

    for i in range(predictions.shape[0] - 1):
        # have we made any money this time step?
        profit = check_position(profit, actual[i] - actual[i-1])

        curr = actual[i]
        future = predictions[i+1:i+Param.n_out + 1]

        window = np.concatenate(curr, future)

        for j in range(1, Param.n_out + 1):
            if window[j] - window[j-1] > 0:
                is_full = add_signal(x, i+j-1, Signal.LONG)
            elif window[j] - window[j-1] < 0:
                is_full = add_signal(x, i+j-1, Signal.SHORT)
            else:
                pass
            # END if

            if is_full:
                d[i+j-1] = make_trade(x, i+j-1, actual[i+j-2], predictions[i+j-1])
                if not np.isinf(d[i+j-1]):
                    Signal.current_pos = d[i+j-1]
                # END if
            # END if
        # END for
    # END for
    return profit
            
def add_signal(x, t, signal):
    is_full = False
    for i in range(Param.n_out):
        if np.isinf(x[t, i]):
            x[t, i] = signal
            if i == Param.n_out - 1:
                is_full = True
            # END if
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

    if np.abs((predicted - recent_actual)/recent_actual) > 0.01:
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