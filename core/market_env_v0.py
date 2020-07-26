import numpy as np
from collections import deque
import pandas as pd
import math

from .reward_functions import PercentChange, SimpleProfit, RiskAdjustedReturns

class MarketEnvironmentV0():
    '''
    Simple market environment, in which the agent could only buy or sell
    '''
    def __init__(self, x, actions, state_size, 
                start_index=0, end_index=10_000,
                undo_detrend=[], close_col=0, predicted_col=4,
                initial_cash=20.0, commission=0, maximum_loss=1.0, inaction_penalty=-1e-05, large_holdings_penalty=-1e-05, 
                max_possible_holdings=20, lost_all_cash_penalty=-1e2, profit_window_size=10, no_trades_penalty=-1e2,
                reward_function='RiskAdjustedReturns', close_padding=1):
        self.start_index = start_index
        self.end_index = end_index
        self.x = x[self.start_index:self.end_index]
        self.undo_detrend = undo_detrend[self.start_index: self.end_index + 1]
        self.detrended = len(undo_detrend) > 0
        self.num_steps = self.x.shape[0]

        self.close_col = close_col
        self.predicted_col = predicted_col
        self.initial_cash = initial_cash
        self.commission = commission
        self.maximum_loss = maximum_loss
        self.inaction_penalty = inaction_penalty
        self.max_possible_holdings = max_possible_holdings
        self.large_holdings_penalty = large_holdings_penalty
        self.lost_all_cash_penalty = lost_all_cash_penalty
        self.no_trades_penalty = no_trades_penalty
        self.close_padding = close_padding


        # Actions
        self.actions = actions
        
        # State size
        self.state_size = state_size

        # Reward function
        if reward_function == 'PercentChange': 
            self.reward_function = PercentChange(window_size=profit_window_size)
        elif reward_function == 'SimpleProfit': 
            self.reward_function = SimpleProfit(window_size=profit_window_size)
        elif reward_function == 'RiskAdjustedReturns': 
            self.reward_function = RiskAdjustedReturns(window_size=profit_window_size, num_steps=self.num_steps)

        self.profit_window_size = profit_window_size

        self.reset()


    def get_action_size(self):
        return len(self.actions)

    def get_state_size(self):
        return self.state_size

    def step(self, action, debug=False):
        '''
        Take one step at the environment
        '''
        if debug: print('-' * 50)
        state = self.get_state(self.step_value)
        if debug: print(">> State: ", state)
        reward = self.make_action(state, action, debug)
        done = self.check_done()
        
        self.step_value += 1
        next_state = self.get_state(self.step_value)

        return next_state, reward, done, None











    #
    # HELPERS
    #

    def get_info(self):
        '''
        Log all the info from the environment
        '''
        return {
            "cash_profit": str(((self.portfolio_value + self.cash) / self.initial_cash - 1) * 100.0),
            "cash": str(self.cash),
            "portfolio_value": str(self.portfolio_value),
            "net": str(self.portfolio_value + self.cash),
            "total_profit": str(self.total_profit), 
            "win_trades": str(self.win_trades), 
            "winnning_percent": str(self.win_trades / self.total_trades if self.total_trades > 0 else 0), 
            "total_trades": str(self.total_trades), 
            "holdings": str(self.holdings), 
            "step_value": str(self.step_value),
            "max_holdings": str(self.max_holdings),
            "total_longs": str(self.total_longs),
            "total_shorts": str(self.total_shorts),
            "win_longs": str(self.win_longs / self.total_longs if self.total_longs > 0 else 0),
            "win_shorts": str(self.win_shorts / self.total_shorts if self.total_shorts > 0 else 0),
        }


    def get_state(self, step=0):
        state = np.array(self.x[step])
        return state

    def next_state(self):
        return self.get_state(self.step_value + 1)


    def reset(self):
        """
        Resets the game, clears the state buffer.
        """
        # State params
        self.holdings = 0

        # Params
        self.inventory = []
        self.total_profit= 0.0
        self.win_trades = 0
        self.total_trades = 0
        self.total_shorts = 0
        self.win_shorts = 0
        self.total_longs = 0
        self.win_longs = 0
        self.step_value = 0
        self.max_holdings = 0
        self.cash = self.initial_cash
        self.portfolio_value = 0
        self.portfolio = { 
            'performance': {
                'net_worth': deque(maxlen=self.profit_window_size)
            }
        }


        return self.get_state(self.step_value)
        

    def make_action(self, state, action, debug=False):
        '''
        Perform an action and change the state
        '''
        close = self.get_last_close()
        penalties = 0
        if debug: print(">> Close: ", close)


        # Update max holdings
        if abs(self.holdings) > self.max_holdings: self.max_holdings = abs(self.holdings)

        # If done
        if self.step_value == self.num_steps - 1:
            if self.total_trades == 0: penalties += self.no_trades_penalty
            # Se terminou calculo o lucro final
            if self.holdings < 0: self._close_short(close, debug)
            if self.holdings > 0: self._close_long(close, debug)

        # Sit
        if(action == 0):
            if debug: print("Sitting...")
            penalties += self.inaction_penalty

        # Buy
        elif (action == 1):
            # Buy
            if close < self.cash and abs(self.holdings) < self.max_possible_holdings:
                self.inventory.insert(0, close)
                self.holdings += 1
                self.cash -= close
                if debug: print("LONG: $", '%.4f' % close, " - Holdings: ", self.holdings)

        # Sell
        elif(action == 2):
            # CLOSE LONG
            if self.holdings > 0: 
                self._close_all_long(close, debug)

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


    def get_last_close(self):
        if self.detrended: return self.undo_detrend[self.step_value + self.close_padding][self.close_col]
        return self.x[self.step_value][self.close_col]


    def _lost_all_cash(self, debug=False):
        # Lost all money
        if self.cash <= self.maximum_loss * self.initial_cash - self.initial_cash: 
            if debug: print(">> LOST ALL CASH: No money")
            return True
        # Short more than expected
        if np.sum(self.inventory) < 0 and np.sum(self.inventory) <= -self.maximum_loss * self.initial_cash: 
            if debug: print(">> LOST ALL CASH: Too many shorts")
            return True
        return False

    def check_done(self, debug=False):
        ''' 
        Check if we get to the end of the run
        '''
        if self._lost_all_cash(): return True
        return self.step_value == self.num_steps - 2


    def _close_long(self, close, debug=False):
        last_position = self.inventory.pop(0)
        profit = close - last_position - self.commission
        self.cash += profit + last_position
        self.holdings -= 1
        self.total_profit += profit

        # Add to- total trades
        self.total_trades += 1
        self.total_longs +=1 
        if profit > 0: 
            self.win_trades += 1
            self.win_longs += 1

        if debug: print("CLOSE ONE LONG: $", '%.4f' % close, 
                    " - Profit $", '%.4f' % profit, 
                    # " - ", '%.4f' % profit_per, "%", 
                    "Total profit", '%.4f' % self.total_profit)

    def _close_all_long(self, close, debug=False):
        profit = self.holdings * close - sum(self.inventory) - self.commission
        self.cash += sum(self.inventory) + profit
        self.inventory = []
        self.holdings = 0
        self.total_profit += profit

        # Add to- total trades
        self.total_trades += 1
        self.total_longs +=1 
        if profit > 0: 
            self.win_trades += 1
            self.win_longs += 1

        if debug: print("CLOSE LONG: $", '%.4f' % close, 
                    " - Profit $", '%.4f' % profit, 
                    # " - ", '%.4f' % profit_per, "%", 
                    "Total profit", '%.4f' % self.total_profit)

    def _update_portfolio(self):
        '''
        Update our portfolio value (essential to calculate the reward)
        '''
        net_worth = self.portfolio_value + self.cash


        self.portfolio['performance']['net_worth'].append(net_worth)
        self.portfolio['performance']['win_percent'] = self.win_trades / self.total_trades if \
            self.total_trades > 0 else 0

        if len(self.portfolio['performance']['net_worth']) > self.profit_window_size:
            self.portfolio['performance']['net_worth'].popleft()

    def close(self):
      return None
