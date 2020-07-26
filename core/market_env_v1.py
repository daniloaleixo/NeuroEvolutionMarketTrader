import numpy as np
from collections import deque
import pandas as pd
import math
from .market_env_v0 import MarketEnvironmentV0

class MarketEnvironmentV1(MarketEnvironmentV0):
    '''
    Market environment, in which the agent could long or short
    '''
    def __init__(self, x, actions, state_size, 
                start_index=0, end_index=10_000,
                undo_detrend=[], close_col=0, predicted_col=4, seq_len=False,
                initial_cash=20.0, commission=0, maximum_loss=1.0, inaction_penalty=-1e-05, large_holdings_penalty=-1e-05, 
                max_possible_holdings=20, lost_all_cash_penalty=-1e2, profit_window_size=100, no_trades_penalty=-1e2,
                reward_function='RiskAdjustedReturns', close_padding=1):

        MarketEnvironmentV0.__init__(
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

    def make_action(self, state, action, debug=False):
        close = self.get_last_close()
        penalties = 0
        if debug: print(">> Close: ", close)


        # Update max holdings
        if abs(self.holdings) > self.max_holdings: self.max_holdings = abs(self.holdings)

        # If done
        if self.step_value == self.num_steps - 1:
            # Se terminou calculo o lucro final
            if self.holdings < 0: self._close_short(close, debug)
            if self.holdings > 0: self._close_long(close, debug)

        # Sit
        if(action == 0):
            if debug: print("Sitting...")
            penalties += self.inaction_penalty

        # Long
        elif (action == 1):
            # (CLOSE SHORT) If we have previous shorts we sell all of them 
            if self.holdings < 0: self._close_all_short(close, debug)

            # Now long
            if self.holdings >= 0 and close < self.cash and abs(self.holdings) < self.max_possible_holdings:
                self.inventory.insert(0, close)
                self.holdings += 1
                self.cash -= close
                if debug: print("LONG: $", '%.4f' % close, " - Holdings: ", self.holdings)

        # Short
        elif(action == 2):
            # (CLOSE LONG) If we have previous longs we sell all of them 
            if self.holdings > 0: self._close_all_long(close, debug)

            # Now short
            if self.holdings <= 0 and close < self.initial_cash - abs(self.portfolio_value) \
                and abs(self.holdings) < self.max_possible_holdings:
                
                self.inventory.insert(0, -close)
                self.holdings -= 1
                self.cash += close
                if debug: print("SHORT: $", '%.4f' % close, " - Holdings: ", self.holdings)


        # Update portfolio_value and price paid per share
        self.portfolio_value = close * self.holdings

        # Update portfolio info
        self._update_portfolio()

        # Calculate penalties
        lost_all_cash_penalty = self.lost_all_cash_penalty if self._lost_all_cash() else 0
        penalties += lost_all_cash_penalty

        rewards = self.reward_function.get_reward(self.portfolio)

        reward = penalties + rewards
        if debug: print("Reward: ", reward)

        return reward

 

    def _close_all_short(self, close, debug=False):
        profit = -sum(self.inventory) - -self.holdings * close - self.commission
        self.cash += sum(self.inventory) + profit
        self.inventory = []
        self.holdings = 0
        self.total_profit += profit

        # Add to- total trades
        self.total_trades += 1
        self.total_shorts += 1
        if profit > 0: 
            self.win_trades += 1
            self.win_shorts += 1

        if debug: print("CLOSE SHORT: $", '%.4f' % close, 
                    " - Profit $", '%.4f' % profit, 
                    # " - ", '%.4f' % profit_per, "%",
                    "Total profit", '%.4f' % self.total_profit)