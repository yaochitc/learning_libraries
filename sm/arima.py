import tushare as ts
import pandas as pd
import datetime as dt
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt

money_supply = ts.get_money_supply()
money_supply['month'] = money_supply['month'].apply(lambda x:
                                                    dt.datetime.strptime(x,'%Y.%m'))
money_supply[['m2']] = money_supply[['m2']].apply(pd.to_numeric, errors='coerce')

df_m2 = money_supply[['month', 'm2']]

df_m2 = df_m2[::-1]
df_m2 = df_m2.set_index(pd.DatetimeIndex(df_m2['month'], freq='MS'))
df_m2 = df_m2.ix['1996-01-01'::]

size = int(len(df_m2) * 0.66)
train, test = df_m2[0:size], df_m2[size:len(df_m2)]

model = ARIMA(train['m2'], order=(3,1,1))
model_fit = model.fit(disp=0)

output = model_fit.forecast(steps=50)
yhat = output[0]

print(output)

plt.figure(figsize=(10, 8))
plt.plot(df_m2['month'], df_m2['m2'])
plt.ylabel("M2")
plt.xlabel("Month")
plt.show()
