# stock-market-analysis
stock
stock market analysis
In [6]:
In [7]:
In [8]:
In [9]:
In [10]:
In [11]:
Note: you may need to restart the kernel to use updated packages.
ERROR: Could not find a version that satisfies the requirement cuflinks
(from versions: none)
ERROR: No matching distribution found for cuflinks
Out[11]:
Date Open High Low Close Adj Close Volume
0 2019-04-05 196.449997 197.100006 195.929993 197.000000 194.454758 18526600
1 2019-04-08 196.419998 200.229996 196.339996 200.100006 197.514709 25881700
2 2019-04-09 200.320007 202.850006 199.229996 199.500000 196.922470 35768200
3 2019-04-10 198.679993 200.740005 198.179993 200.619995 198.027985 21695300
4 2019-04-11 200.850006 201.000000 198.440002 198.949997 196.379578 20900800
pip install cuflinks
import numpy as np
import pandas as pd
import warnings
import cufflinks as cf
import plotly.graph_objects as go
from plotly.offline import iplot, init_notebook_mode
import matplotlib.pyplot as plot
import seaborn as sns
import io
import requests
import datetime
import mplfinance as mpf
Apple_df=pd.read_csv("AAPL.csv")
Apple_df.head()
In [12]:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 252 entries, 0 to 251
Data columns (total 7 columns):
# Column Non-Null Count Dtype
--- ------ -------------- -----
0 Date 252 non-null object
1 Open 252 non-null float64
2 High 252 non-null float64
3 Low 252 non-null float64
4 Close 252 non-null float64
5 Adj Close 252 non-null float64
6 Volume 252 non-null int64
dtypes: float64(5), int64(1), object(1)
memory usage: 13.9+ KB
Apple_df.info()
In [13]:
Out[13]:
Date Open High Low Close Adj Close Volume
222 2020-02-24 297.260010 304.179993 289.230011 298.179993 298.179993 55548800
223 2020-02-25 300.950012 302.529999 286.130005 288.079987 288.079987 57668400
224 2020-02-26 286.529999 297.880005 286.500000 292.649994 292.649994 49513700
225 2020-02-27 281.100006 286.000000 272.959991 273.519989 273.519989 80151400
226 2020-02-28 257.260010 278.410004 256.369995 273.359985 273.359985 106721200
227 2020-03-02 282.279999 301.440002 277.720001 298.809998 298.809998 85349300
228 2020-03-03 303.670013 304.000000 285.799988 289.320007 289.320007 79868900
229 2020-03-04 296.440002 303.399994 293.130005 302.739990 302.739990 54794600
230 2020-03-05 295.519989 299.549988 291.410004 292.920013 292.920013 46893200
231 2020-03-06 282.000000 290.820007 281.230011 289.029999 289.029999 56544200
232 2020-03-09 263.750000 278.089996 263.000000 266.170013 266.170013 71686200
233 2020-03-10 277.140015 286.440002 269.369995 285.339996 285.339996 71322500
234 2020-03-11 277.390015 281.220001 271.859985 275.429993 275.429993 63899700
235 2020-03-12 255.940002 270.000000 248.000000 248.229996 248.229996 104618500
236 2020-03-13 264.890015 279.920013 252.949997 277.970001 277.970001 92683000
237 2020-03-16 241.949997 259.079987 240.000000 242.210007 242.210007 80605900
238 2020-03-17 247.509995 257.609985 238.399994 252.860001 252.860001 81014000
239 2020-03-18 239.770004 250.000000 237.119995 246.669998 246.669998 75058400
240 2020-03-19 247.389999 252.839996 242.610001 244.779999 244.779999 67964300
241 2020-03-20 247.179993 251.830002 228.000000 229.240005 229.240005 100423300
242 2020-03-23 228.080002 228.500000 212.610001 224.369995 224.369995 84188200
243 2020-03-24 236.360001 247.690002 234.300003 246.880005 246.880005 71882800
244 2020-03-25 250.750000 258.250000 244.300003 245.520004 245.520004 75900500
245 2020-03-26 246.520004 258.679993 246.360001 258.440002 258.440002 63021800
246 2020-03-27 252.750000 255.869995 247.050003 247.740005 247.740005 51054200
247 2020-03-30 250.740005 255.520004 249.399994 254.809998 254.809998 41994100
248 2020-03-31 255.600006 262.489990 252.000000 254.289993 254.289993 49250500
249 2020-04-01 246.500000 248.720001 239.130005 240.910004 240.910004 44054600
250 2020-04-02 240.339996 245.149994 236.899994 244.929993 244.929993 41483500
251 2020-04-03 242.800003 245.699997 238.970001 241.410004 241.410004 32418200
Apple_df_small=Apple_df[-30:]
Apple_df_small
In [14]:
In [15]:
In [16]:
Out[15]:
Date Open High Low Close Adj Close Volume He
222
2020-
02-24
297.260010 304.179993 289.230011 298.179993 298.179993 55548800 0.919
224
2020-
02-26
286.529999 297.880005 286.500000 292.649994 292.649994 49513700 6.119
226
2020-
02-28
257.260010 278.410004 256.369995 273.359985 273.359985 106721200 16.099
227
2020-
03-02
282.279999 301.440002 277.720001 298.809998 298.809998 85349300 16.529
229
2020-
03-04
296.440002 303.399994 293.130005 302.739990 302.739990 54794600 6.299
Out[16]:
Date Open High Low Close Adj Close Volume Heig
223
2020-
02-25
300.950012 302.529999 286.130005 288.079987 288.079987 57668400 12.87002
225
2020-
02-27
281.100006 286.000000 272.959991 273.519989 273.519989 80151400 7.5800
228
2020-
03-03
303.670013 304.000000 285.799988 289.320007 289.320007 79868900 14.3500
230
2020-
03-05
295.519989 299.549988 291.410004 292.920013 292.920013 46893200 2.5999
234
2020-
03-11
277.390015 281.220001 271.859985 275.429993 275.429993 63899700 1.96002
green_df=Apple_df_small[Apple_df_small.Close>Apple_df_small.Open].copy()
green_df["Height"]=green_df["Close"]-green_df["Open"]
red_df=Apple_df_small[Apple_df_small.Close<Apple_df_small.Open].copy()
red_df["Height"]=red_df["Open"]-red_df["Close"]
green_df.head()
red_df.head()
In [17]:
In [18]:
fig= plot.figure(figsize=(15,7))
plot.vlines(x=Apple_df_small["Date"],ymin=Apple_df_small["Low"],ymax=["High"],
color="grey"
);
#green candles
plot.bar(x=green_df["Date"],height=green_df["Height"],bottom=green_df["Open"],color="g
#Red candles
plot.bar(x=red_df["Date"],height=red_df["Height"],bottom=red_df["Open"],color="oranger
plot.yticks(range(200,340,20),["{} $".format(v) for v in range(200,340,20)]);
plot.xlabel("Date")
plot.ylabel("Price($)")
plot.title("Apple Candlestick Chart");
plot.style.use("fivethirtyeight");
In [19]:
In [20]:
Out[20]:
Date object
Open float64
High float64
Low float64
Close float64
Adj Close float64
Volume int64
dtype: object
fig= plot.figure(figsize=(15,7))
plot.vlines(x=Apple_df_small["Date"],ymin=Apple_df_small["Low"],ymax=["High"],
color="grey"
);
#green candles
plot.bar(x=green_df["Date"],height=green_df["Height"],bottom=green_df["Open"],color="g
#Red candles
plot.bar(x=red_df["Date"],height=red_df["Height"],bottom=red_df["Open"],color="oranger
plot.yticks(range(200,340,20),["{} $".format(v) for v in range(200,340,20)]);
plot.xlabel("Date")
plot.ylabel("Price($)")
plot.title("Apple Candlestick Chart");
Apple_df.dtypes
In [21]:
In [22]:
In [23]:
0.0
Out[22]:
column_name percentage
0 Date 0.0
1 Open 0.0
2 High 0.0
3 Low 0.0
4 Close 0.0
5 Adj Close 0.0
6 Volume 0.0
Out[23]:
<AxesSubplot:>
missing_values_count = Apple_df.isnull().sum()
total_cells = np.product(Apple_df.shape)
total_missing = missing_values_count.sum()
percentage_missing = (total_missing/total_cells)*100
print(percentage_missing)
NAN = [(c, Apple_df[c].isnull().mean()*100) for c in Apple_df]
NAN = pd.DataFrame(NAN, columns=['column_name', 'percentage'])
NAN
sns.set(rc = {'figure.figsize': (20, 5)})
Apple_df['Open'].plot(linewidth = 1,color='blue')
In [24]:
Sort the dataset on date time and filter “Date” and
“Open” columns
cols_plot = ['Open','High','Low','Close']
axes = Apple_df[cols_plot].plot(alpha = 1, figsize=(20, 30), subplots = True)
for ax in axes:
ax.set_ylabel('Variation')
In [25]:
In [26]:
In [27]:
Out[25]:
Date Open High Low Close Adj Close Volume
Date
2019-
04-05
2019-
04-05
196.449997 197.100006 195.929993 197.000000 194.454758 18526600
2019-
04-08
2019-
04-08
196.419998 200.229996 196.339996 200.100006 197.514709 25881700
2019-
04-09
2019-
04-09
200.320007 202.850006 199.229996 199.500000 196.922470 35768200
2019-
04-10
2019-
04-10
198.679993 200.740005 198.179993 200.619995 198.027985 21695300
2019-
04-11
2019-
04-11
200.850006 201.000000 198.440002 198.949997 196.379578 20900800
... ... ... ... ... ... ... ...
2020-
03-30
2020-
03-30
250.740005 255.520004 249.399994 254.809998 254.809998 41994100
2020-
03-31
2020-
03-31
255.600006 262.489990 252.000000 254.289993 254.289993 49250500
2020-
04-01
2020-
04-01
246.500000 248.720001 239.130005 240.910004 240.910004 44054600
2020-
04-02
2020-
04-02
240.339996 245.149994 236.899994 244.929993 244.929993 41483500
2020-
04-03
2020-
04-03
242.800003 245.699997 238.970001 241.410004 241.410004 32418200
252 rows × 7 columns
Out[27]:
Open float64
High float64
Low float64
Close float64
Adj Close float64
Volume int64
dtype: object
Apple_df["Date"]=pd.to_datetime(Apple_df.Date,format="%Y-%m-%d")
Apple_df.index=Apple_df['Date']
Apple_df
del Apple_df["Date"]
Apple_df.dtypes
In [28]:
In [29]:
Out[28]:
Open High Low Close Adj Close Volume
Date
2019-04-05 NaN NaN NaN NaN NaN NaN
2019-04-08 NaN NaN NaN NaN NaN NaN
2019-04-09 NaN NaN NaN NaN NaN NaN
2019-04-10 NaN NaN NaN NaN NaN NaN
2019-04-11 NaN NaN NaN NaN NaN NaN
2019-04-12 NaN NaN NaN NaN NaN NaN
2019-04-15 198.642857 200.272860 197.477140 199.181427 196.608010 2.400999e+07
2019-04-16 199.072859 200.882858 197.852855 199.502856 196.925289 2.503424e+07
2019-04-17 199.518572 201.332859 198.177142 199.935713 197.352557 2.546640e+07
2019-04-18 199.918570 201.518572 198.647143 200.558570 197.967368 2.381320e+07
Out[29]:
<AxesSubplot:xlabel='Date'>
Apple_df.rolling(7).mean().head(10)
Apple_df['Open'].plot(figsize=(20,8),alpha = 1)
Apple_df.rolling(window=30).mean()['Close'].plot(alpha = 1)
In [30]:
Optional specify a minimum numbe2of periods
In [31]:
Out[30]:
<AxesSubplot:xlabel='Date'>
Out[31]:
<AxesSubplot:xlabel='Date'>
Apple_df['Close: 30 Day Mean'] = Apple_df['Close'].rolling(window=30).mean()
Apple_df[['Close','Close: 30 Day Mean']].plot(figsize=(20,8),alpha = 1)
Apple_df['Close'].expanding(min_periods=1).mean().plot(figsize=(20,8),alpha = 1)
In [32]:
In [33]:
LSTM are sensitive to the scale of the data. so we
apply MinMax scaler
Out[32]:
0 196.449997
1 196.419998
2 200.320007
3 198.679993
4 200.850006
...
247 250.740005
248 255.600006
249 246.500000
250 240.339996
251 242.800003
Name: Open, Length: 252, dtype: float64
Out[33]:
[<matplotlib.lines.Line2D at 0x1fcd31d6760>]
Apple_df=Apple_df.reset_index()['Open']
Apple_df
plot.plot(Apple_df)
In [34]:
splitting dataset into train and test split
In [35]:
In [36]:
[[0.14072335]
[0.14052242]
[0.16664439]
[0.1556597 ]
[0.17019428]
[0.15914265]
[0.15498997]
[0.16088417]
[0.16141991]
[0.18539849]
[0.18345614]
[0.19417276]
[0.21379773]
[0.21024784]
[0.19732079]
[0.19397183]
[0.18499664]
[0.23067653]
[0.23040855]
[0 23744139]
Out[36]:
(189, 63)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
Apple_df=scaler.fit_transform(np.array(Apple_df).reshape(-1,1))
print(Apple_df)
train_size=int(len(Apple_df)*0.75)
test_size=len(Apple_df)-train_size
train_data,test_data=Apple_df[0:train_size,:],Apple_df[train_size:len(Apple_df),:1]
train_size,test_size
In [37]:
convert an array of values into a dataset matrix¶
In [38]:
reshape into X=t,t+1,t+2,t+3 and Y=t+4
In [39]:
In [40]:
Out[37]:
(array([[0.14072335],
[0.14052242],
[0.16664439],
[0.1556597 ],
[0.17019428],
[0.15914265],
[0.15498997],
[0.16088417],
[0.16141991],
[0.18539849],
[0.18345614],
[0.19417276],
[0.21379773],
[0.21024784],
[0.19732079],
[0.19397183],
[0.18499664],
[0.23067653],
(88, 100)
(88,)
Out[40]:
(None, None)
train_data,test_data
def create_dataset(dataset, time_step=1):
train_X, train_Y = [], []
for i in range(len(dataset)-time_step-1):
a = dataset[i:(i+time_step), 0] ###i=0, 0,1,2,3-----99 100
train_X.append(a)
train_Y.append(dataset[i + time_step, 0])
return numpy.array(train_X), numpy.array(train_Y)
import numpy
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)
print(X_train.shape), print(y_train.shape)
In [ ]:
In [ ]:
In [ ]:
In [ ]:
In [ ]:
In [ ]:[stock market analysis - Jupyter Notebook.pdf](https://github.com/Lochanitin/stock-market-analysis/files/12182490/stock.market.analysis.-.Jupyter.Notebook.pdf)
