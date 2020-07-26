
  

# Neuro Evolution Market Trader
Neuro evolution agent to trade

**If you like this project you can support me.**  
<a href="https://www.buymeacoffee.com/daniloaleixo" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-white.png" alt="Buy Me A Coffee" style="height: 51px !important;width: 217px !important;" ></a>

It took me several experiments to get to this agent. I tried several deep learning architectures and technical analysis parameters. 

## Agent v 3.0141

The agent has the following characteristics:
* Using neuro evolution
* Receives OHLC as parameters
* Generates RSI and MACD technical indicators to be used as parameters
* Also run through a pretrained CNN buy and sell classifier to get the last set of parameters  


## Results
Right now the best agent [v3.0141](https://github.com/daniloaleixo/NeuroEvolutionMarketTrader#runinfo-v30141---ohlc--cnn-classifier-bs-w10-15x15--rsimacd), the agent uses OHLC, RSI and MACD inputs, as well as the output of a pretrained CNN buy and sell classifier. 
The agent has these results applied to EURUSD with 30M candles:
* Mean over backtest returns: **14.17%**
* Points above buy and hold mean: **12.26%**
* Winning Percent mean: **80%**

### Training
![img](https://github.com/daniloaleixo/NeuroEvolutionMarketTrader/blob/master/images/Screenshot%20from%202020-07-25%2021-17-43.png)

*Max fitness for each generation*

![img](https://github.com/daniloaleixo/NeuroEvolutionMarketTrader/blob/master/images/Screenshot%20from%202020-07-25%2021-17-49.png)

*Mean fitness for each generation*

### Backtesting
#### 2016 15M candles
![img](https://github.com/daniloaleixo/NeuroEvolutionMarketTrader/blob/master/images/Screenshot%20from%202020-07-25%2022-20-36.png)
#### 2017 30M candles
![img](https://github.com/daniloaleixo/NeuroEvolutionMarketTrader/blob/master/images/Screenshot%20from%202020-07-25%2021-27-19.png)
#### 2018 30M candles
![img](https://github.com/daniloaleixo/NeuroEvolutionMarketTrader/blob/master/images/Screenshot%20from%202020-07-25%2021-27-51.png)

### RUNINFO: v3.0141 - OHLC + CNN Classifier B/S w10 15x15 + RSI&MACD
##### Params 
- layers:			16, 32, 64, 32, 16
- population_size: 256
- generations: 100
- episodes: 10
- mutation_variance: 0.005
- survival_ratio: 0.3
- both_parent_percentage: 0.8
- one_parent_percentage: 0.1 
- reward_function: SimpleProfit
- initial_cash: 5.0
- profit_window_size: 10
- close_col: 0
- large_holdings_penalty: 0
- lost_all_cash_penalty: -1e2
- inaction_penalty: 0
##### Results
* After 100 generations
  * Max rewards: 2.508929
  * Mean rewards: 0.963821
  * Std rewrads: 1.400484  
  * Best Profit (Env 10k - 50k): 37.23
  * Mean profit over all best genome (Env 10k - 50k):  14.35
* Backtesting:
  * Mean of 2016 + 2017 + 2018 returns: **14.17**
  * Points above BH mean: **12.26**
  * Winning Percent mean: 80%
  * 2016 15M
    * Return [%] 					 24.8711
    * Buy & Hold Return [%]  				 -3.144688847131627
    * Max. Drawdown [%] 				 2.58
    * Avg. Drawdown [%] 				 0.64
    * n Trades 					 7860
    * Win Rate [%] 					 81.40
    * Best Trade [%] 					 0.12363999999999997
    * Worst Trade [%]  				 -0.08284999999999965
    * SQN 						 4.67
    * Sharpe Ratio 					 0.0
    * Sortino Ratio 					 -2.1676
  * 2017 30M
    * Return [%] 					 8.5952
    * Buy & Hold Return [%]  				 13.862653618570775
    * Max. Drawdown [%] 				 3.55
    * Avg. Drawdown [%] 				 0.75
    * n Trades 					 3160
    * Win Rate [%] 					 80.54
    * Best Trade [%] 					 0.02414999999999967
    * Worst Trade [%]  				 -0.04445000000000032
    * SQN 						 2.20
    * Sharpe Ratio 					 0.0
    * Sortino Ratio 					 -2.3002
  * 2018 30M
    * Return [%] 					 9.0453
    * Buy & Hold Return [%]  				 -5.008444957273516
    * Max. Drawdown [%] 				 2.72
    * Avg. Drawdown [%] 				 0.12
    * n Trades 					 3318
    * Win Rate [%] 					 78.99
    * Best Trade [%] 					 0.024970000000000603
    * Worst Trade [%]  				 -0.038909999999999556
    * SQN 						 2.31
    * Sharpe Ratio 					 0.0
    * Sortino Ratio 					 -2.8040

