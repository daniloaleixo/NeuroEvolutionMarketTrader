import backtrader as bt

debug = True

# Create a Strategy
class BaseStrategy(bt.Strategy):
    params = dict(
        stop_loss=0.002,  # price is 2% less than the entry point
    )
    

    def log(self, txt, dt=None):
        if debug:
            ''' Logging function fot this strategy'''
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.count = 0
        self.last_order = None

    def buy_with_stop_loss(self, size=1):
        self.buy_order = self.buy(size=size, transmit=False)

        # Setting parent=buy_order ... sends both together
        stop_price = self.data.close[0] * (1.0 - self.p.stop_loss)
        self.sell(exectype=bt.Order.Stop, price=stop_price, parent=self.buy_order)
        self.log('STOP LOSS CREATED, Price: %.4f' % stop_price)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            self.last_order = order
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.4f, Cost: %.4f, Comm %.4f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.4f, Cost: %.4f, Comm %.4f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed: return

        self.log('OPERATION PROFIT, GROSS %.4f, NET %.4f' % (trade.pnl, trade.pnlcomm))