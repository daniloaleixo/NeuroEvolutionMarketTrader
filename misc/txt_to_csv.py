import sys
import pandas as pd

usage =  '''
Transform the dataframe I have in txt to a csv that I will use inside the modules

python txt_to_csv.py <input.txt> <output.csv>

'''

if (len(sys.argv) < 3):
  print(usage)
  exit(-1)

def correct_time(x):
    x = str(x)
    if len(x) == 6: return x[:2] + ":" + x[2:4] + ":" + x[4:]
    if len(x) == 5: return "0" + x[:1] + ":" + x[1:3] + ":" + x[3:]
    if len(x) == 4: return "00:" + x[:2] + ":" + x[2:]
    if len(x) == 3: return "00:" + "0" + x[:1] + ":" + x[1:]
    if len(x) == 1: return "00:00:00"

def correct_year(x):
  x = str(x)
  return x[:4] + "-" + x[4:6] + "-" + x[6:]

data = pd.read_csv(sys.argv[1], sep=",")
data["<TIME>"] = data["<TIME>"].apply(correct_time)
data["<DTYYYYMMDD>"] = data["<DTYYYYMMDD>"].apply(correct_year)
data["Datetime"] = data["<DTYYYYMMDD>"].astype(str) + " " + data["<TIME>"].astype(str)
data = data.set_index("Datetime")
data = data.rename(columns={"<OPEN>": "Open", "<HIGH>": "High", "<LOW>": "Low", "<CLOSE>": "Close"})
data = data.drop(['<TICKER>', '<DTYYYYMMDD>', '<TIME>', '<VOL>'], axis=1)
data.to_csv(sys.argv[2])  