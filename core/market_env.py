import numpy as np
from collections import deque
import pandas as pd
import math

from .reward_functions import SimpleProfit, RiskAdjustedReturns

class MarketEnvironment(object):
    """  
    Market Environment
    """
    def __init__(self, x, actions, state_size, 
                start_index=0, end_index=10_000,
                undo_detrend=[], close_col=0, predicted_col=4, seq_len=False,
                initial_cash=20.0, commission=0, maximum_loss=1.0, inaction_penalty=-1e-05, large_holdings_penalty=-1e-05, 
                max_possible_holdings=20, lost_all_cash_penalty=-1e2, profit_window_size=100,
                reward_function='RiskAdjustedReturns'):
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


        # Actions
        self.actions = actions
        
        # State size
        self.state_size = state_size

        self.seq_len = seq_len
        self.state_buffer = deque(maxlen=self.seq_len)

        # Reward function
        if reward_function == 'SimpleProfit': 
            self.reward_function = SimpleProfit(window_size=profit_window_size)
        elif reward_function == 'ProfitAndWinningTrades': 
            self.reward_function = ProfitAndWinningTrades(window_size=profit_window_size)
        elif reward_function == 'RiskAdjustedReturns': 
            self.reward_function = RiskAdjustedReturns(window_size=profit_window_size, num_steps=self.num_steps)
        elif reward_function == 'RiskAdjustedReturnsAndWinningPercentage': 
            self.reward_function = RiskAdjustedReturnsAndWinningPercentage(window_size=profit_window_size, num_steps=self.num_steps)

        self.profit_window_size = profit_window_size

        self.reset()


    def get_info(self):
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
        state = np.insert(state, self.predicted_col + 1, self.holdings) # Holdings
        state = np.insert(state, self.predicted_col + 2, self.cash / self.initial_cash) # Cash 
        state = np.insert(state, self.predicted_col + 3, self.portfolio_value / self.initial_cash) # portfolio value 
        state = np.insert(state, self.predicted_col + 4, self.avg_price_per_share) # average price paid per share  
        return np.expand_dims(state, axis=1)
            
    def get_action_size(self):
        return len(self.actions)

    def get_state_size(self):
        return self.state_size
    
    def get_last_close(self):
        if self.detrended: return self.undo_detrend[self.step_value + 1][self.close_col]
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
        if self._lost_all_cash(): return True
        return self.step_value == self.num_steps - 2

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
        self.avg_price_per_share = 0
        self.portfolio = { 
            'performance': {
                'net_worth': deque(maxlen=self.profit_window_size)
            }
        }


        # When sequence Fills the buffer
        if self.seq_len:
            for i in range(0, self.seq_len + 1): self.state_buffer.append(self.get_state(i + 1))
            
            self.step_value = self.seq_len + 1
            return np.array(self.state_buffer).squeeze()

        else: 
            return self.get_state(self.step_value)
        

    
    def reset_one(self): return self.reset().ravel()
        
    def next_state(self):
        return self.get_state(self.step_value + 1)
        
    def make_action(self, state, action, debug=False):
        close = self.get_last_close()
        penalties = 0
        if debug: print(">> Close: ", close)


        # Update max holdings
        if abs(self.holdings) > self.max_holdings: self.max_holdings = abs(self.holdings)

        # If done
        if self.step_value == self.num_steps - 1:
            # Se terminou calculo o lucro final
            if sum(self.inventory) < 0: self._close_short(close, debug)
            if sum(self.inventory) > 0: self._close_long(close, debug)

        # Sit
        # elif(action == 0):
        #     if debug: print("Sitting...")
        #     penalties += self.inaction_penalty

        # Long
        elif (action == 0):
            # (CLOSE SHORT) If we have previous shorts we sell all of them 
            if sum(self.inventory) < 0: self._close_short(close, debug)

            # Now long
            if self.holdings == 0:
                self.inventory.insert(0, close)
                self.holdings += 1
                self.cash -= close
                if debug: print("LONG: $", '%.4f' % close, " - Holdings: ", self.holdings)

        # Short
        elif(action == 1):
            # (CLOSE LONG) If we have previous longs we sell all of them 
            if sum(self.inventory) > 0: self._close_long(close, debug)

            # Now short
            if self.holdings == 0:
                self.inventory.insert(0, -close)
                self.holdings -= 1
                self.cash += close
                if debug: print("SHORT: $", '%.4f' % close, " - Holdings: ", self.holdings)




        # Update portfolio_value and price paid per share
        self.portfolio_value = close * self.holdings
        self.avg_price_per_share = close - np.sum(self.inventory) / self.holdings if not self.holdings == 0 else 0

        # Update portfolio info
        self._update_portfolio()

        # Calculate penalties
        lost_all_cash_penalty = self.lost_all_cash_penalty if self._lost_all_cash() else 0
        penalties += lost_all_cash_penalty

        rewards = self.reward_function.get_reward(self.portfolio)


        reward = penalties + rewards
        if debug: print("Reward: ", reward)

        return reward
        
    def step(self, action, debug=False):

        state = self.get_state(self.step_value)
        # print('&&&& curr state', state)
        reward = self.make_action(state, action, debug)

        done = self.check_done()
        
        self.step_value += 1
        next_state = self.get_state(self.step_value)
        if self.seq_len:
            self.state_buffer.popleft()
            self.state_buffer.append(next_state)
            next_state = np.array(self.state_buffer).squeeze()


        return next_state, reward, done, None
    
    def step_one(self, action, debug=False): 
        next_state, reward, done, _ = self.step(action, debug)
        return next_state.ravel(), reward, done, None


    def _close_short(self, close, debug=False):
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

    def _close_long(self, close, debug=False):
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
        net_worth = self.portfolio_value + self.cash


        self.portfolio['performance']['net_worth'].append(net_worth)
        self.portfolio['performance']['win_percent'] = self.win_trades / self.total_trades if \
            self.total_trades > 0 else 0

        if len(self.portfolio['performance']['net_worth']) > self.profit_window_size:
            self.portfolio['performance']['net_worth'].popleft()

    def close(self): return None