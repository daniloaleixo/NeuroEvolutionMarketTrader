import backtrader as bt
import numpy as np

class ExtendedPandasData(bt.feeds.PandasData):

    # lines = (
    #     "SMA",
    #     "BBANDS_up",
    #     "BBANDS_md",
    #     "BBANDS_dw",
    #     "EMA",
    #     "HT_TRENDLINE",
    #     "WMA",

    #     "ADX",
    #     "MACD",
    #     "MOM",
    #     "RSI",
    #     # "OBV",
    #     "ATR",
    #     "TRANGE",
    #     "AVGPRICE",
    #     "MEDPRICE",

    #     "WCLPRICE",
    #     "LINEARREG_SLOPE",
    #     "STDDEV",

    # )
    
    # params = (
    #   ("SMA", "SMA"),
    #   ("BBANDS_up", "BBANDS_up"),
    #   ("BBANDS_md", "BBANDS_md"),
    #   ("BBANDS_dw", "BBANDS_dw"),
    #   ("EMA", "EMA"),
    #   ("HT_TRENDLINE", "HT_TRENDLINE"),
    #   ("WMA", "WMA"),
    #   ("ADX","ADX"),
    #     ("MACD","MACD"),
    #     ("MOM","MOM"),
    #     ("RSI","RSI"),
    #     # ("OBV","OBV"),
    #     ("ATR","ATR"),
    #     ("TRANGE","TRANGE"),
    #     ("AVGPRICE","AVGPRICE"),
    #     ("MEDPRICE","MEDPRICE"),
    #     ("WCLPRICE","WCLPRICE"),
    #     ("LINEARREG_SLOPE","LINEARREG_SLOPE"),
    #     ("STDDEV","STDDEV"),
    # )
    
    def get_array_cols(self, cols, size):
        info = np.zeros((size, len(cols)))
        for i in range(0, len(cols)):
            col = cols[i]
            if col == "Close": info[:, i] = self.close.get(size=size)
            if col == "Open": info[:, i] = self.open.get(size=size)
            if col == "High": info[:, i] = self.high.get(size=size)
            if col == "Low": info[:, i] = self.low.get(size=size)
            if col == "Volume": info[:, i] = self.volume.get(size=size)
            # if col == "SMA": info[:, i] = self.SMA.get(size=size)
            # if col == "BBANDS_up": info[:, i] = self.BBANDS_up.get(size=size)
            # if col == "BBANDS_md": info[:, i] = self.BBANDS_md.get(size=size)
            # if col == "BBANDS_dw": info[:, i] = self.BBANDS_dw.get(size=size)
            # if col == "EMA": info[:, i] = self.EMA.get(size=size)
            # if col == "HT_TRENDLINE": info[:, i] = self.HT_TRENDLINE.get(size=size)
            # if col == "WMA": info[:, i] = self.WMA.get(size=size)
            # if col == "ADX": info[:, i] = self.ADX.get(size=size)
            # if col == "MACD": info[:, i] = self.MACD.get(size=size)
            # if col == "MOM": info[:, i] = self.MOM.get(size=size)
            # if col == "RSI": info[:, i] = self.RSI.get(size=size)
            # # if col == "OBV": info[:, i] = self.OBV.get(size=size)
            # if col == "ATR": info[:, i] = self.ATR.get(size=size)
            # if col == "TRANGE": info[:, i] = self.TRANGE.get(size=size)
            # if col == "AVGPRICE": info[:, i] = self.AVGPRICE.get(size=size)
            # if col == "MEDPRICE": info[:, i] = self.MEDPRICE.get(size=size)
            # if col == "WCLPRICE": info[:, i] = self.WCLPRICE.get(size=size)
            # if col == "LINEARREG_SLOPE": info[:, i] = self.LINEARREG_SLOPE.get(size=size)
            # if col == "STDDEV": info[:, i] = self.STDDEV.get(size=size)
        return info
        