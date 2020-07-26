

from abc import abstractmethod
import numpy as np
import math


class RewardScheme:
    def reset(self):
        """Optionally implementable method for resetting stateful schemes."""
        pass

    def get_reward(self, portfolio):
        """
        Arguments:
            portfolio: The portfolio being used by the environment.
        Returns:
            A float corresponding to the benefit earned by the action taken this timestep.
        """
        raise NotImplementedError()

class PercentChange(RewardScheme):
    """A simple reward scheme that rewards the agent for incremental increases in net worth."""

    def __init__(self, window_size = 10):
        self.window_size = window_size
        print("PercentChange")

    def reset(self):
        pass

    def get_reward(self, portfolio):
        """Rewards the agent for incremental increases in net worth over a sliding window.
        Args:
            portfolio: The portfolio being used by the environment.
        Returns:
            The cumulative percentage change in net worth over the previous `window_size` timesteps.
        """
        if len(portfolio['performance']['net_worth']) < 1: return 0
        
        pct_change = np.diff(portfolio['performance']['net_worth']) / np.array(portfolio['performance']['net_worth'])[1:]
        pct_change = pct_change[~np.isnan(pct_change)] # Drop nan
        return pct_change[-1] if len(pct_change) > 0 else 0


class SimpleProfit(RewardScheme):
    """A simple reward scheme that rewards the agent for cumulative incremental increases in net worth."""

    def __init__(self, window_size = 10):
        self.window_size = window_size
        print("SimpleProfit")

    def reset(self):
        pass

    def get_reward(self, portfolio):
        """Rewards the agent for incremental increases in net worth over a sliding window.
        Args:
            portfolio: The portfolio being used by the environment.
        Returns:
            The cumulative percentage change in net worth over the previous `window_size` timesteps.
        """
        if len(portfolio['performance']['net_worth']) < 1: return 0
        
        pct_change = np.diff(portfolio['performance']['net_worth']) / np.array(portfolio['performance']['net_worth'])[1:]
        pct_change = pct_change[~np.isnan(pct_change)] # Drop nan
        returns = np.cumprod(1.0 + pct_change) - 1 # Cumulative prod
        return returns[-1] if len(returns) > 0 else 0


class RiskAdjustedReturns(RewardScheme):
    """A reward scheme that rewards the agent for increasing its net worth, while penalizing more volatile strategies.
    """

    def __init__(self, window_size = 10, num_steps = 1, _risk_free_rate = 0, _target_returns = 0, _reduction_factor=1e3):
        self.window_size = window_size
        self._risk_free_rate = _risk_free_rate
        self._target_returns = _target_returns
        self._reduction_factor = _reduction_factor
        self.num_steps = num_steps
        print("RiskAdjustedReturns")

    def reset(self):
        pass

    def get_reward(self, portfolio):
        """Rewards the agent for incremental in sortino ratio
        Args:
            portfolio: The portfolio being used by the environment.
        Returns:
            The cumulative percentage change in net worth and the winning streak over the previous `window_size` timesteps.
        """
        if len(portfolio['performance']['net_worth']) < 1: return 0

        pct_change = np.diff(portfolio['performance']['net_worth']) / np.array(portfolio['performance']['net_worth'])[1:]
        returns = pct_change[~np.isnan(pct_change)] # Drop nan

        # Sortino
        if len(returns) > 0:
            was_down = returns < self._target_returns
            downside_returns = []
            for i in range(0, len(was_down)):
                if was_down[i]:
                    downside_returns.append(returns[i] ** 2)
            expected_return = np.mean(returns)
            downside_std = np.sqrt(np.std(downside_returns))
            sortino_ratio = (expected_return - self._risk_free_rate + 1e-9) / (downside_std + 1e-9)
            sortino_ratio = 0 if math.isnan(sortino_ratio) else sortino_ratio

            return sortino_ratio / (self.num_steps * self._reduction_factor)
        
        return 0
