import numpy as np
from collections import deque
import pandas as pd
import math
from .market_env_v1 import MarketEnvironmentV1

class MarketEnvironmentV2(MarketEnvironmentV1):
    '''
    Market environment, in which the agent could long or short
    And the state in slightly different, because I'm adding portoflio info
    '''
    def __init__(self, x, actions, state_size, 
                start_index=0, end_index=10_000,
                undo_detrend=[], close_col=0, predicted_col=4, seq_len=False,
                initial_cash=20.0, commission=0, maximum_loss=1.0, inaction_penalty=-1e-05, large_holdings_penalty=-1e-05, 
                max_possible_holdings=20, lost_all_cash_penalty=-1e2, profit_window_size=100, no_trades_penalty=-1e2,
                reward_function='RiskAdjustedReturns', close_padding=1):

        MarketEnvironmentV1.__init__(
          self,
          x,
          actions,
          state_size,
          start_index=start_index,
          end_index=end_index,
          undo_detrend=undo_detrend,
          close_col=close_col,
          predicted_col=predicted_col,
          initial_cash=initial_cash,
          commission=commission,
          maximum_loss=maximum_loss,
          inaction_penalty=inaction_penalty,
          large_holdings_penalty=large_holdings_penalty,
          max_possible_holdings=max_possible_holdings,
          lost_all_cash_penalty=lost_all_cash_penalty,
          profit_window_size=profit_window_size,
          reward_function=reward_function,
          close_padding=close_padding
        )


    
    def get_state(self, step=0):
        '''
        Adding holdings and avg holdings price diff from actual price
        '''
        state = np.array(self.x[step])
        # Holdings
        state = np.insert(state, state.shape[0], self.holdings)
        # Actual close - average portfolio price
        state = np.insert(
          state, 
          state.shape[0],
          self.get_last_close() / (np.sum(self.inventory) / self.holdings) - 1.0 if self.holdings != 0 else 0 
        )
        return state