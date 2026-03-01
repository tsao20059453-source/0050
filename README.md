```python

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_excel('0050_歷史股價與成交量.xlsx')

data = data[data["成交量"] > 0].copy()

for numbers in [1,2,3,5,10,20]:
    data[f"trade_before{numbers}"] = data["成交量"].shift(numbers)

data["trade_fivemean"] = data["成交量"].rolling(5).mean()

data["trade_tenmean"] = data["成交量"].rolling(10).mean()

data["price_tenmean"] = data["開盤價"].rolling(10).mean()

data["highprice_tenmean"] = data["最高價"].rolling(10).mean()

data["lowprice_tenmean"] = data["最低價"].rolling(10).mean()

data["trade_5std"] = data["成交量"].rolling(5).std()

data["tradechangerate"] = data["成交量"].pct_change()

data["pricechange"] = data["開盤價"].pct_change()

data["highpricechange"] = data["最高價"].pct_change()

data["lowpricechange"] = data["最低價"].pct_change()

data["trade_10dayago_ratio"] = data["成交量"] / data["trade_tenmean"]

data["trade_log"] = np.log1p(data["成交量"])

data["closeprize_1"] = data["收盤價"].shift(1)

data = data.dropna().reset_index(drop = True)


data.head()




features = [
    "trade_before1","trade_before2","trade_before3",
    "trade_before5","trade_before10","trade_before20",
    "成交量","trade_fivemean","trade_tenmean","trade_5std",
    "tradechangerate","trade_10dayago_ratio","trade_log","closeprize_1",
    "開盤價","最高價","最低價","pricechange","highpricechange","lowpricechange",
    "price_tenmean","highprice_tenmean","lowprice_tenmean"
]

X = data[features]
y = data["收盤價"]

splt_idx = int(len(data)*0.8)

X_train = X.iloc[:splt_idx]
X_valid = X.iloc[splt_idx:]

y_train = y.iloc[:splt_idx]
y_valid = y.iloc[splt_idx:]

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

model_1 = Ridge(alpha=0.7, random_state = 0)
model_1.fit(X_train,y_train)

predict = model_1.predict(X_valid)

mae_1 = mean_absolute_error(y_valid, predict)

rmse_1 = np.sqrt(mean_squared_error(y_valid, predict))



model_2 = RandomForestRegressor(
    n_estimators=300,
    max_depth=4,
    min_samples_leaf=5,
    random_state=0,
)

model_2.fit(X_train, y_train)

predict_2 = model_2.predict(X_valid)

mae_2 = mean_absolute_error(y_valid, predict_2)

rmse_2 = np.sqrt(mean_squared_error(y_valid, predict_2))
print(mae_1, rmse_1)

print(mae_2, rmse_2)
from collections import deque

class TradeVolumePredictor:

    def __init__(self, model, featurecol, init_last20_trade):

        self.model = model

        self.featurecol = featurecol

        self.hist = deque([float(a) for a in init_last20_trade], maxlen=21)

    def build_features(self, srs20: np.ndarray) -> np.ndarray:

        a = srs20[-1]
        prea = srs20[-2]

        feature = {
            "trade_before1": srs20[-2],
            "trade_before2": srs20[-3],
            "trade_before3": srs20[-4],
            "trade_before5": srs20[-6],
            "trade_before10": srs20[-11],
            "trade_before20": srs20[-21],
            "trade_fivemean": srs20[-5:].mean(),
            "trade_tenmean": srs20[-10:].mean(),
            "trade_5std": srs20[-5:].std(ddof=0),
            "trade_log": np.log1p(a),
            "closeprize_1": data["收盤價"].iloc[-2]
        }

        feature["開盤價"]= data["開盤價"].iloc[-1]
        feature["最高價"]= data["最高價"].iloc[-1]
        feature["最低價"]= data["最低價"].iloc[-1]
        feature["price_tenmean"]= data["price_tenmean"].iloc[-1]
        feature["highprice_tenmean"]= data["highprice_tenmean"].iloc[-1]
        feature["lowprice_tenmean"]= data["lowprice_tenmean"].iloc[-1]
        feature["lowpricechange"]= data["lowpricechange"].iloc[-1]
        feature["pricechange"]= data["pricechange"].iloc[-1]
        feature["highpricechange"]= data["highpricechange"].iloc[-1]
        feature["成交量"] = a
        feature["tradechangerate"] = (a-prea) / prea 
        feature["trade_10dayago_ratio"] = a / feature["trade_tenmean"]
        
        x = np.array([[feature[c] for c in self.featurecol]], dtype=np.float64)

        return x
    
    def predict(self,tradetoday: float) -> float:
        a = float(tradetoday)
        if not np.isfinite(a) or a<0 :
            raise ValueError("輸入必須大於０")
        
        close_1 = float(data["收盤價"].iloc[-2])
        
        self.hist.append(a)

        srs20 = np.array(self.hist, dtype= np.float64)
        x = self.build_features(srs20)

        return float(self.model.predict(x)[0])

    
init_last20 = data["成交量"].iloc[-20:].to_list()

tvp = TradeVolumePredictor(model_1, features, init_last20)

%pip install ipywidgets

import ipywidgets as widgets
from IPython.display import display, clear_output

trade_box = widgets.IntText(description="成交量")
btn = widgets.Button(description="Predict")
out = widgets.Output()

def on_click(_):
    with out:
        clear_output()
        pred = tvp.predict(trade_box.value)   
        print(f"預測收盤價：{pred:.2f}")

btn.on_click(on_click)
display(trade_box, btn, out)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

dates = pd.to_datetime(data["日期"].iloc[splt_idx:]).to_numpy()

fig, ax = plt.subplots(figsize = (12,4))
ax.plot(dates, y_valid, label="Actual")
ax.plot(dates, predict, label="Predicted")

loc = mdates.AutoDateLocator(maxticks=20, minticks=5)

ax.xaxis.set_major_locator(loc)

ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

fig.tight_layout()
plt.xlabel("dates")
plt.ylabel("close")
plt.title(f"Test: Actual vs Predicted (MAE={mae_1:.3f}, RMSE={rmse_1:.3f})")
plt.legend()
plt.show()

print(dates.dtype)
import shap 

data_for_prediction = X_valid.iloc[[-1]] 
explainer = shap.Explainer(model_1, X_train)

shap_values = explainer(data_for_prediction)

shap.initjs()
shap.plots.force(shap_values[0])
shap.plots.waterfall(shap_values[0], max_display=20)

```
