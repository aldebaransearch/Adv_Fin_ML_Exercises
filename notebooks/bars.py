
import numpy as np
from enum import Enum
from numba import jit


class TYPE(Enum):
    TICK = 1
    VOLUME = 2
    DOLLAR = 3


def _calculate_bars(in_tick_data, add_open_close_info, group_by_index):

    tick_data = in_tick_data.copy()
    tick_data['sign_size'] = np.where(tick_data['side'] == 'Buy', tick_data['homeNotional'],
                                      -tick_data['homeNotional'])

    bars = tick_data['price'].groupby(group_by_index, axis=0).ohlc()
    bars['volume'] = tick_data['homeNotional'].groupby(group_by_index, axis=0).sum()
    bars['count_ticks'] = tick_data['homeNotional'].groupby(group_by_index, axis=0).count()
    bars['trade_flow_imbalance'] = tick_data['sign_size'].groupby(group_by_index, axis=0).sum()
    bars['dollar'] = (tick_data.price * tick_data.homeNotional).groupby(group_by_index, axis=0).sum()

    if add_open_close_info:
        close_index = np.concatenate([np.argwhere(np.diff(group_by_index)).ravel(),[-1]])
        open_index = np.concatenate([[0], close_index[:-1] + 1])

        bars['open_time'] = tick_data.index[open_index]
        bars['close_time'] = tick_data.index[close_index]
        bars['open_trdMatchID'] = tick_data['trdMatchID'].iloc[open_index].values
        bars['close_trdMatchID'] = tick_data['trdMatchID'].iloc[close_index].values

    return bars

def tick(data, n_tick_pr_event, add_open_close_info = True):

    condition = np.arange(len(data.index))
    return _calculate_bars(data, add_open_close_info, condition // n_tick_pr_event)


def volume(data, volume, add_open_close_info = True):

    condition = np.cumsum(data.homeNotional)
    return _calculate_bars(data, add_open_close_info, condition // volume)


def dollar(data, amount, add_open_close_info = True):

    condition = np.cumsum(data.price * data.homeNotional)
    return _calculate_bars(data, add_open_close_info, condition // amount)


@jit(nopython=True)
def _imbalance(property, initial_imbalance, alpha):

    weighted_count = 0 # Denominator for normalization of EWMAs
    weighted_sum_T = 0 # Nominator for EWMA of duration of a bar
    weighted_sum_imbalance = 0 # Nominator for EWMA of the imbalance of bar

    out = np.zeros(property.shape)
    dummy = 0
    imbalance = initial_imbalance
    T = 0

    for i in range(len(property)):
        dummy += property[i]
        T += 1
        if (abs(dummy)>=imbalance):
            out[i] = 1
            weighted_sum_T = T + (1-alpha)*weighted_sum_T
            weighted_sum_imbalance = dummy/(1.0*T) + (1 - alpha) * weighted_sum_imbalance
            weighted_count = 1 + (1-alpha) * weighted_count
            ewma_T = weighted_sum_T/weighted_count
            ewma_imbalance = weighted_sum_imbalance/weighted_count
            imbalance = ewma_T * abs(ewma_imbalance)
            dummy = 0
            T = 0

    out[0] = 1

    return np.cumsum(out) # Cumsum'ing to accomodate groupby o pandas dataframe in _calculate_bars


def tick_imbalance(data, initial_imbalance, alpha, add_open_close_info = True):

    out = _calculate_bars(data, add_open_close_info, _imbalance(np.where(data['side'] == 'Buy', 1, -1),initial_imbalance,alpha))

    return out


def volume_imbalance(data, initial_imbalance, alpha, add_open_close_info = True):

    out = _calculate_bars(data, add_open_close_info, _imbalance((np.where(data['side'] == 'Buy', 1, -1) * data.homeNotional).values,initial_imbalance,alpha))

    return out


def dollar_imbalance(data, initial_imbalance, alpha, add_open_close_info = True):

    out = _calculate_bars(data, add_open_close_info, _imbalance((np.where(data['side'] == 'Buy', 1, -1) * (data.homeNotional*data.price)).values, initial_imbalance, alpha))

    return out


@jit(nopython=True)
def _runs(property, initial_run_length, alpha):

    weighted_count = 0 # Denominator for normalization of EWMAs
    weighted_sum_T = 0 # Nominator for EWMA of duration of a bar
    weighted_sum_up = 0 # Nominator for EWMA of the proportions of up-ticks

    out = np.zeros(property.shape)
    dummy_up = 0
    run_length = initial_run_length
    T = 0

    for i in range(len(property)):
        if property[i]>0:
            dummy_up += 1
        T += 1
        dummy = max(dummy_up,T-dummy_up)
        if (abs(dummy) >= run_length):
            out[i] = 1
            weighted_sum_T = T + (1 - alpha) * weighted_sum_T
            weighted_sum_up = dummy_up / (1.0 * T) + (1 - alpha) * weighted_sum_up
            weighted_count = 1 + (1 - alpha) * weighted_count
            ewma_T = weighted_sum_T / weighted_count
            ewma_up = weighted_sum_up / weighted_count
            run_length = ewma_T * max(ewma_up,1-ewma_up)
            dummy_up = 0
            T = 0

    out[0] = 1

    return np.cumsum(out) # Cumsum'ing to accomodate groupby o pandas dataframe in _calculate_bars


@jit(nopython=True)
def _runs_vol_dollar(property, initial_run_size, alpha):

    weighted_count = 0 # Denominator for normalization of EWMAs
    weighted_sum_T = 0 # Nominator for EWMA of duration of a bar
    weighted_sum_up = 0 # Nominator for EWMA of the proportions of up-ticks
    weighted_sum_property_up = 0 # Nominator for EWMA of either volume or dollar (hence named 'property') amount up
    weighted_sum_property_down = 0 # Nominator for EWMA of either volume or dollar (hence named 'property') amount down

    out = np.zeros(property.shape)
    dummy_up = 0
    property_up = 0
    property_down = 0
    run_size = initial_run_size
    T = 0
    dummy_count = 0

    for i in range(len(property)):
        if property[i]>0:
            dummy_up += 1
            property_up += property[i]
        else:
            property_down += -1*property[i]
        T += 1
        dummy = max(property_up,property_down)
        if (dummy >= run_size):
            out[i] = 1
            weighted_sum_T = T + (1 - alpha) * weighted_sum_T
            weighted_sum_up = dummy_up / (1.0 * T) + (1 - alpha) * weighted_sum_up
            weighted_sum_property_up = (property_up/dummy_up if dummy_up else 0) + (1 - alpha) * weighted_sum_property_up
            weighted_sum_property_down = (property_down/(T-dummy_up) if (T-dummy_up) else 0) + (1 - alpha) * weighted_sum_property_down
            weighted_count = 1 + (1 - alpha) * weighted_count
            ewma_T = weighted_sum_T / weighted_count
            ewma_up = weighted_sum_up / weighted_count
            ewma_property_up = weighted_sum_property_up / weighted_count
            ewma_property_down = weighted_sum_property_down / weighted_count
            run_size = ewma_T * max(ewma_up * ewma_property_up,(1-ewma_up) * ewma_property_down)
            dummy_up = 0
            property_up = 0
            property_down = 0
            T = 0
            dummy_count += 1

    out[0] = 1

    return np.cumsum(out) # Cumsum'ing to accomodate groupby o pandas dataframe in _calculate_bars


def tick_runs(data, initial_run_length, alpha, add_open_close_info = True):

    runs =_runs(np.where(data['side'] == 'Buy', 1, -1), initial_run_length, alpha)
    out = _calculate_bars(data, add_open_close_info, runs)

    return out


def volume_runs(data, initial_run_length, alpha, add_open_close_info = True):

    runs = _runs_vol_dollar((np.where(data['side'] == 'Buy', 1, -1) * data.homeNotional).values,initial_run_length,alpha)
    out = _calculate_bars(data, add_open_close_info, runs)

    return out


def dollar_runs(data, initial_run_length, alpha, add_open_close_info = True):

    runs = _runs_vol_dollar((np.where(data['side'] == 'Buy', 1, -1) * (data.homeNotional*data.price)).values, initial_run_length, alpha)
    out = _calculate_bars(data, add_open_close_info, runs)

    return out


if __name__ == '__main__':

    # I am using it like this since I have a really large file of ticks and uses the final bars.
    # Maybe someone can find inspiration. Otherwise code below gives something that more resembles
    # the nomenclature of the book

    # import pandas as pd
    # store = pd.HDFStore('path_to_file.h5')
    # nrows = store.get_storer('data').nrows
    #
    # chunk_size = 1000000
    #
    # for i in range(nrows // chunk_size + 1):
    #     print('Processing {0} out of {1} steps'.format(i+1,nrows // chunk_size + 1))
    #     chunk = store.select('data',
    #                          start=i * chunk_size,
    #                          stop=(i + 1) * chunk_size)
    #
    #     if i > 0:
    #         chunk = rest.append(chunk)
    #
    #     temp_bars = dollar_runs(chunk, 100000 , 0.01)
    #
    #     rest = chunk.iloc[-1 * temp_bars.iloc[-1].count_ticks:]
    #
    #     if i == 0:
    #         result = temp_bars.iloc[:-1]
    #     else:
    #         result = result.append(temp_bars.iloc[:-1])

    import pandas as pd

    data = pd.read_csv() # Load from your prefered method
    b_t = np.where(data['side'] == 'Buy', 1, -1) # I am so lucky as to get the information from the exchange
    runs = _runs_vol_dollar((b_t * (data.homeNotional * data.price)).values, 100000, 0.01)