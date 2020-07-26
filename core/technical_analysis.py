import numpy
import talib
from tqdm import tqdm

def add_technical_indicators(dataframe):

  # Overlap Studies Functions
  dataframe["SMA"] = talib.SMA(dataframe["Close"])
  dataframe["BBANDS_up"], dataframe["BBANDS_md"], dataframe["BBANDS_dw"] = talib.BBANDS(dataframe["Close"], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
  dataframe["EMA"]  = talib.EMA(dataframe["Close"], timeperiod=30)
  dataframe["HT_TRENDLINE"] = talib.HT_TRENDLINE(dataframe["Close"])
  dataframe["WMA"] = talib.WMA(dataframe["Close"], timeperiod=30)

  # Momentum Indicator Functions
  dataframe["ADX"] = talib.ADX(dataframe["High"], dataframe["Low"], dataframe["Close"], timeperiod=14)
  dataframe["MACD"], _, _ = talib.MACD(dataframe["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
  dataframe["MOM"] = talib.MOM(dataframe["Close"], timeperiod=5)
  dataframe["RSI"] = talib.RSI(dataframe["Close"], timeperiod=14)

  # Volume Indicator Functions
  # dataframe["OBV"] = talib.OBV(dataframe["Close"], dataframe["Volume"])

  # Volatility Indicator Functions
  dataframe["ATR"] = talib.ATR(dataframe["High"], dataframe["Low"], dataframe["Close"], timeperiod=14)
  dataframe["TRANGE"] = talib.TRANGE(dataframe["High"], dataframe["Low"], dataframe["Close"])

  # Price Transform Functions
  dataframe["AVGPRICE"] = talib.AVGPRICE(dataframe["Open"], dataframe["High"], dataframe["Low"], dataframe["Close"])
  dataframe["MEDPRICE"] = talib.MEDPRICE(dataframe["High"], dataframe["Low"])
  dataframe["WCLPRICE"] = talib.WCLPRICE(dataframe["High"], dataframe["Low"], dataframe["Close"])

  # Statistic Functions
  dataframe["LINEARREG_SLOPE"] = talib.LINEARREG_SLOPE(dataframe["Close"], timeperiod=14)
  dataframe["STDDEV"] = talib.STDDEV(dataframe["Close"], timeperiod=5, nbdev=1)






  dataframe = dataframe.dropna()
  return dataframe


def add_technical_indicators_with_intervals(df, indicators=[], intervals=[], true_value=False, verbose=False):

  indicators_range = range(len(indicators)) if not verbose else tqdm(range(len(indicators)))
  for i in indicators_range:
    indicator = indicators[i]

    for interval in intervals:
      
      # 1  SMA
      if indicator == 'SMA': 
        if not true_value:
          df["SMA_"+str(interval)] = df["Close"] - talib.SMA(df["Close"], timeperiod=interval)
        else:
          df["SMA_"+str(interval)] = talib.SMA(df["Close"], timeperiod=interval)

      # 2 KAMA
      if indicator == 'KAMA': 
        if not true_value:
          df["KAMA_"+str(interval)] = df["Close"] - talib.KAMA(df["Close"], timeperiod=interval)
        else:
          df["KAMA_"+str(interval)] = talib.KAMA(df["Close"], timeperiod=interval)

      # 3 MIDPRICE
      if indicator == 'MIDPRICE': 
        if not true_value:
          df["MIDPRICE_"+str(interval)] = df["Close"] - talib.MIDPRICE(df["High"], df["Low"], timeperiod=interval)
        else:
          df["MIDPRICE_"+str(interval)] = talib.MIDPRICE(df["High"], df["Low"], timeperiod=interval)

      # 4 MIDPOINT
      if indicator == 'MIDPOINT': 
        if not true_value:
          df["MIDPOINT_"+str(interval)] = df["Close"] - talib.MIDPOINT(df["Close"], timeperiod=interval)
        else:
          df["MIDPOINT_"+str(interval)] = talib.MIDPOINT(df["Close"], timeperiod=interval)

      # 5 EMA
      if indicator == 'EMA': 
        if not true_value:
          df["EMA_"+str(interval)] = df["Close"] - talib.EMA(df["Close"], timeperiod=interval)
        else:
          df["EMA_"+str(interval)] = talib.EMA(df["Close"], timeperiod=interval)

      # 6 DEMA
      if indicator == 'DEMA': 
        if not true_value:
          df["DEMA_"+str(interval)] = df["Close"] - talib.DEMA(df["Close"], timeperiod=interval)
        else:
          df["DEMA_"+str(interval)] = talib.DEMA(df["Close"], timeperiod=interval)

      # 7 TEMA
      if indicator == 'TEMA': 
        if not true_value:
          df["TEMA_"+str(interval)] = df["Close"] - talib.TEMA(df["Close"], timeperiod=interval)
        else:
          df["TEMA_"+str(interval)] = talib.TEMA(df["Close"], timeperiod=interval)
      
      # 8 TRIMA
      if indicator == 'TRIMA': 
        if not true_value:
          df["TRIMA_"+str(interval)] = df["Close"] - talib.TRIMA(df["Close"], timeperiod=interval)
        else:
          df["TRIMA_"+str(interval)] = talib.TRIMA(df["Close"], timeperiod=interval)

      # 9 WMA
      if indicator == 'WMA': 
        if not true_value:
          df["WMA_"+str(interval)] = df["Close"] - talib.WMA(df["Close"], timeperiod=interval)
        else:
          df["WMA_"+str(interval)] = talib.WMA(df["Close"], timeperiod=interval)

      # 10 LINEARREG
      if indicator == 'LINEARREG': 
        if not true_value:
          df["LINEARREG_"+str(interval)] = df["Close"] - talib.LINEARREG(df["Close"], timeperiod=interval)
        else:
          df["LINEARREG_"+str(interval)] = talib.LINEARREG(df["Close"], timeperiod=interval)

      # 11 TSF
      if indicator == 'TSF': 
        if not true_value:
          df["TSF_"+str(interval)] = df["Close"] - talib.TSF(df["Close"], timeperiod=interval)
        else:
          df["TSF_"+str(interval)] = talib.TSF(df["Close"], timeperiod=interval)

      # 12  SMAO
      if indicator == 'SMAO': 
        if not true_value:
          df["SMAO_"+str(interval)] = df["Close"] - talib.SMA(df["Open"], timeperiod=interval)
        else:
          df["SMAO_"+str(interval)] = talib.SMA(df["Open"], timeperiod=interval)

      # 13 KAMAO
      if indicator == 'KAMAO': 
        if not true_value:
          df["KAMAO_"+str(interval)] = df["Close"] - talib.KAMA(df["Open"], timeperiod=interval)
        else:
          df["KAMAO_"+str(interval)] = talib.KAMA(df["Open"], timeperiod=interval)

      # 14 MIDPOINTO
      if indicator == 'MIDPOINTO': 
        if not true_value:
          df["MIDPOINTO_"+str(interval)] = df["Close"] - talib.MIDPOINT(df["Open"], timeperiod=interval)
        else:
          df["MIDPOINTO_"+str(interval)] = talib.MIDPOINT(df["Open"], timeperiod=interval)

      # 15 EMAO
      if indicator == 'EMAO': 
        if not true_value:
          df["EMAO_"+str(interval)] = df["Close"] - talib.EMA(df["Open"], timeperiod=interval)
        else:
          df["EMAO_"+str(interval)] = talib.EMA(df["Open"], timeperiod=interval)

      # 16 DEMAO
      if indicator == 'DEMAO': 
        if not true_value:
          df["DEMAO_"+str(interval)] = df["Close"] - talib.DEMA(df["Open"], timeperiod=interval)
        else:
          df["DEMAO_"+str(interval)] = talib.DEMA(df["Open"], timeperiod=interval)

      # 17 TEMAO
      if indicator == 'TEMAO': 
        if not true_value:
          df["TEMAO_"+str(interval)] = df["Close"] - talib.TEMA(df["Open"], timeperiod=interval)
        else:
          df["TEMAO_"+str(interval)] = talib.TEMA(df["Open"], timeperiod=interval)


      # 18 TRIMAO
      if indicator == 'TRIMAO': 
        if not true_value:
          df["TRIMAO_"+str(interval)] = df["Close"] - talib.TRIMA(df["Open"], timeperiod=interval)
        else:
          df["TRIMAO_"+str(interval)] = talib.TRIMA(df["Open"], timeperiod=interval)

      # 19 WMAO
      if indicator == 'WMAO': 
        if not true_value:
          df["WMAO_"+str(interval)] = df["Close"] - talib.WMA(df["Open"], timeperiod=interval)
        else:
          df["WMAO_"+str(interval)] = talib.WMA(df["Open"], timeperiod=interval)

      # 20 LINEARREGO
      if indicator == 'LINEARREGO': 
        if not true_value:
          df["LINEARREGO_"+str(interval)] = df["Close"] - talib.LINEARREG(df["Open"], timeperiod=interval)
        else:
          df["LINEARREGO_"+str(interval)] = talib.LINEARREG(df["Open"], timeperiod=interval)

      # 21 TSFO
      if indicator == 'TSFO': 
        if not true_value:
          df["TSFO_"+str(interval)] = df["Close"] - talib.TSF(df["Open"], timeperiod=interval)
        else:
          df["TSFO_"+str(interval)] = talib.TSF(df["Open"], timeperiod=interval)

      # 22 MACD
      if indicator == 'MACD' or indicator == 'MACDhist': 
        df["MACD_"+str(interval)], _, df["MACDhist_"+str(interval)] = talib.MACD(df["Close"], fastperiod=interval, slowperiod=interval * 2, signalperiod=int(interval * 1.5))

      # 23 MACDFIX
      if indicator == 'MACDFIX': 
        df["MACDFIX_"+str(interval)], _, _ = talib.MACDFIX(df["Close"], signalperiod=interval)

      # 24 MOM
      if indicator == 'MOM': 
        df["MOM_"+str(interval)] = talib.MOM(df["Close"], timeperiod=interval)

      # 25 ROCP
      if indicator == 'ROCP': 
        df["ROCP_"+str(interval)] = talib.ROCP(df["Close"], timeperiod=interval)
      
      # 26 APO
      if indicator == 'APO': 
        df["APO_"+str(interval)] = talib.APO(df["Close"], fastperiod=interval, slowperiod=interval*2)

      # 27 MINUS_DM
      if indicator == 'MINUS_DM': 
        df["MINUS_DM_"+str(interval)] = talib.MINUS_DM(df["High"], df["Low"], timeperiod=interval)

      # 28 PLUS_DM
      if indicator == 'PLUS_DM': 
        df["PLUS_DM_"+str(interval)] = talib.PLUS_DM(df["High"], df["Low"], timeperiod=interval)
      
      # 29 BETA
      if indicator == 'BETA': 
        df["BETA_"+str(interval)] = talib.BETA(df["High"], df["Low"], timeperiod=interval) / 100.0

      # 30 TRIX
      if indicator == 'TRIX': 
        df["TRIX_"+str(interval)] = talib.TRIX(df["Close"], timeperiod=interval)

      # 31 ATR
      if indicator == 'ATR': 
        df["ATR_"+str(interval)] = talib.ATR(df["High"], df["Low"], df["Close"], timeperiod=interval)

      # 32 PPO
      if indicator == 'PPO': 
        df["PPO_"+str(interval)] = talib.PPO(df["Close"], fastperiod=interval, slowperiod=interval * 2)

      # 33 RSI
      if indicator == 'RSI': 
        df["RSI_"+str(interval)] = talib.RSI(df["Close"], timeperiod=interval) / 100.0
      
      # 34 RSIO
      if indicator == 'RSIO': 
        df["RSIO_"+str(interval)] = talib.RSI(df["Open"], timeperiod=interval) / 100.0
      
      # 35 LINEARREG_ANGLE
      if indicator == 'LINEARREG_ANGLE': 
        df["LINEARREG_ANGLE_"+str(interval)] = talib.LINEARREG_ANGLE(df["Close"], timeperiod=interval) / 100.0

      # 36 MINUS_DI
      if indicator == 'MINUS_DI': 
        df["MINUS_DI_"+str(interval)] = talib.MINUS_DI(df["High"], df["Low"], df["Close"], timeperiod=interval) / 100.0

      # 37 PLUS_DI
      if indicator == 'PLUS_DI': 
        df["PLUS_DI_"+str(interval)] = talib.PLUS_DI(df["High"], df["Low"], df["Close"], timeperiod=interval) / 100.0

      # 38 DX
      if indicator == 'DX': 
        df["DX_"+str(interval)] = talib.DX(df["High"], df["Low"], df["Close"], timeperiod=interval) / 100.0

      # 39 ADX
      if indicator == 'ADX': 
        df["ADX_"+str(interval)] = talib.ADX(df["High"], df["Low"], df["Close"], timeperiod=interval) / 100.0

      # 40 ADXO
      if indicator == 'ADXO': 
        df["ADXO_"+str(interval)] = talib.ADX(df["High"], df["Low"], df["Open"], timeperiod=interval) / 100.0

      # 41 ADXR
      if indicator == 'ADXR': 
        df["ADXR_"+str(interval)] = talib.ADXR(df["High"], df["Low"], df["Close"], timeperiod=interval) / 100.0

      # 42 CCI
      if indicator == 'CCI': 
        df["CCI_"+str(interval)] = talib.CCI(df["High"], df["Low"], df["Close"], timeperiod=interval) /100.0

      # 42 CCIO
      if indicator == 'CCIO': 
        df["CCIO_"+str(interval)] = talib.CCI(df["High"], df["Low"], df["Open"], timeperiod=interval) /100.0

      # 43 WILLR
      if indicator == 'WILLR': 
        df["WILLR_"+str(interval)] = talib.WILLR(df["High"], df["Low"], df["Close"], timeperiod=interval) / 100.0

      # 44 WILLRO
      if indicator == 'WILLRO': 
        df["WILLRO_"+str(interval)] = talib.WILLR(df["High"], df["Low"], df["Open"], timeperiod=interval) / 100.0

      # 45 CMO
      if indicator == 'CMO': 
        df["CMO_"+str(interval)] = talib.CMO(df["Close"], timeperiod=interval) / 100.0

      # 46 AROONOSC
      if indicator == 'AROONOSC': 
        df["AROONOSC_"+str(interval)] = talib.AROONOSC(df["High"], df["Low"], timeperiod=interval) / 100.0

      # 47 CORREL
      if indicator == 'CORREL': 
        df["CORREL_"+str(interval)] = talib.CORREL(df["High"], df["Low"], timeperiod=interval)



  return df.dropna()
