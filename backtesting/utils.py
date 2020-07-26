import numpy as np

def print_analysis(data, thestrat, final_cash, start_date, end_date, divider = "\t", 
        _risk_free_rate = 0.01, _target_returns = 0):
    first_price =  data[-(len(data) - 1)]
    last_price = data[0]
    trade_analysis = thestrat.analyzers.mytradeanalysis.get_analysis()
    return_analysis = thestrat.analyzers.myreturnanalysis.get_analysis()
    dd_analysis = thestrat.analyzers.myddanalysis.get_analysis()
    sqn_analysis = thestrat.analyzers.mysqnanalysis.get_analysis()
    sr_analysis = thestrat.analyzers.mysranalysis.get_analysis()
    all_returns_analysis = thestrat.analyzers.mypyanalysis.get_analysis()['returns']


    # Calculate sortino ratio
    returns = np.array(list(all_returns_analysis.values()))
    if len(returns) > 0:
        was_down = returns < _target_returns
        downside_returns = []
        for i in range(0, len(was_down)):
            if was_down[i]:
                downside_returns.append(returns[i] ** 2)
        expected_return = np.mean(returns)
        downside_std = np.sqrt(np.std(downside_returns))
        sortino_ratio = (expected_return - _risk_free_rate + 1e-9) / (downside_std + 1e-9)
    else: sortino_ratio = 0

    # print("Start", divider, start_date.strftime("%Y-%m-%d"))
    # print("End", divider, end_date.strftime("%Y-%m-%d"))
    print("Duration", divider * 5 + str((end_date - start_date).days), "days")
    print("Bars length", divider * 5 + str(trade_analysis['len']['total']) if 'len' in trade_analysis else "-" )
    print("Equity Final [$]", divider * 4,  '%.4f' % final_cash)
    print("Equity Peak [$]", divider * 5)
    print("Return [%]", divider * 5, '%.4f' % (return_analysis["rtot"] * 100))
    print("Buy & Hold Return [%] ", divider * 4, ((last_price / first_price) - 1.0) *100)
    print("Max. Drawdown [%]", divider * 4, '%.2f' % dd_analysis["max"]["drawdown"])
    print("Avg. Drawdown [%]", divider * 4, '%.2f' % dd_analysis["drawdown"])
    print("# Trades", divider * 5, trade_analysis["total"]["total"])
    print("Win Rate [%]", divider * 5, '%.2f' % (100.0 * trade_analysis["won"]["total"] / trade_analysis["total"]["total"]) if 'won' in trade_analysis else "-")
    print("Best Trade [%]", divider * 5, trade_analysis["won"]["pnl"]["max"] if 'won' in trade_analysis else "-")
    print("Worst Trade [%] ", divider * 4, trade_analysis["lost"]["pnl"]["max"] if 'lost' in trade_analysis else "-")
    print("SQN", divider * 6, '%.2f' % sqn_analysis["sqn"])
    print("Sharpe Ratio", divider * 5, '%.4f' % sr_analysis["sharperatio"] if sr_analysis["sharperatio"] else 0.0)
    print("Sortino Ratio", divider * 5, '%.4f' % sortino_ratio)
    print("_strategy", divider * 5, "test strategy")


def normalise_windows(window_data, single_window=False):
    '''Normalise window with a base value of zero'''
    normalised_data = []
    window_data = [window_data] if single_window else window_data
    for window in window_data:
        normalised_window = []
        for col_i in range(window.shape[1]):
            normalised_col = [((float(p) / (float(window[0, col_i]) + 0.00000001) ) - 1) for p in window[:, col_i]]
            normalised_window.append(normalised_col)
        normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
        normalised_data.append(normalised_window)
    return np.array(normalised_data)

def normalize_to_first_element(array):
    new_arr = array / array[0] - 1.0
    res = np.zeros(array.shape)
    res[0] = array[0]
    res[1:] = new_arr[1:]
    return res