import pandas as pd
import numpy as np
import talib as ta
from sklearn import linear_model as lm
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.model_selection import ParameterGrid

data = pd.read_csv('TSLA.csv')
data['Date'] = pd.to_datetime(data['Date'], format='%Y/%m/%d')
data.set_index('Date', inplace=True)
for i in data.index:
    if i > datetime(2019,1,1):
        data = data.drop([i])

data['5d_future_close'] = data['Adj_Close'].shift(-5)
data['5d_close_future_pct'] = data['5d_future_close'].pct_change(5)
data['5d_close_pct'] = data['Adj_Close'].pct_change(5)
data['HL_pct'] = (data['High']-data['Low'])/data['Low']
data['Volume_1d_change'] = data['Volume'].pct_change(1)
feature_names = ['5d_close_pct', 'Volume_1d_change']
for n in [14, 30, 200]:
    data['sma' + str(n)] = ta.SMA(data['Adj_Close'].values, timeperiod=n)
    data['rsi' + str(n)] = ta.RSI(data['Adj_Close'].values, timeperiod=n)
    data['ema' + str(n)] = ta.EMA(data['Adj_Close'].values, timeperiod=n)
    feature_names = feature_names + ['rsi' + str(n), 'ema' + str(n)]
data['ATR'] = ta.ATR(data['High'].values, data['Low'].values, data['Adj_Close'].values, timeperiod=14)
data['ADX'] = ta.ADX(data['High'].values, data['Low'].values, data['Adj_Close'].values, timeperiod=14)
macd, macdsignal, macdhist = ta.MACD(data['Adj_Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
data['MACD'] = macd
data['MACDsignal'] = macdsignal
feature_names = feature_names +['ATR', 'ADX', 'MACD', 'MACDsignal']
data.tail()
data = data.dropna()

################################################################
features = data[feature_names]
targets = data['Adj_Close']
train_size = int(0.8 * targets.shape[0])
train_features = features[:train_size]
train_targets = targets[:train_size]
test_features = features[train_size:]
test_targets = targets[train_size:]

skmodel = lm.LinearRegression().fit(train_features,train_targets)
def get_skmodel(skmodel, train_features, train_targets ,test_features, test_targets):
    print('R^2 score on train=', skmodel.score(train_features, train_targets))
    print('R^2 score on test=', skmodel.score(test_features, test_targets))
    print('intercept = ', skmodel.intercept_, '\n', 'slope=', skmodel.coef_, '\n')
    train_predictions = skmodel.predict(train_features)
    test_predictions = skmodel.predict(test_features)
    plt.scatter(train_targets, train_predictions, label='train', alpha=0.4, color='b')
    plt.scatter(test_targets, test_predictions, label='test', alpha=0.4, color='orangered')
    #plt.scatter(targets, targets, label='Original', alpha=0.3, color='navajowhite')
    # Plot the perfect prediction line
    xmin, xmax = plt.xlim()
    plt.plot(np.arange(xmin, xmax, 0.01), np.arange(xmin, xmax, 0.01), c='k')
    plt.xlabel('predictions')
    plt.ylabel('actual')
    plt.legend()
    plt.show()

get_skmodel(skmodel,train_features, train_targets, test_features, test_targets)

###############################  RFR


rfr = RandomForestRegressor(n_estimators=200) #I used a loop to select hyperparameters but almost similar answer....
rfr.fit(train_features, train_targets)
print(rfr.score(train_features, train_targets))
print(rfr.score(test_features, test_targets))

####Loop through the parameter grid, set the hyperparameters, and save the scores
#grid = {'n_estimators':[i for i in range(20,201)], 'max_depth': [3,4], 'max_features': [8], 'random_state': [42]}
#test_scores = []
#for g in ParameterGrid(grid):
    #rfr.set_params(**g)  # ** is "unpacking" the dictionary
    #rfr.fit(train_features, train_targets)
    #test_scores.append(rfr.score(test_features, test_targets))
#best_idx = np.argmax(test_scores)
#print('best score and para \n', test_scores[best_idx], ParameterGrid(grid)[best_idx])


#Print one tree
t = RandomForestRegressor(n_estimators=5, max_depth=3).fit(test_features, test_targets)
Tree = t.estimators_[1]
plt.figure()
tree.plot_tree(Tree,filled=True, rounded=True, fontsize=4);
plt.show()
#Prediction
train_predictions = rfr.predict(train_features)
test_predictions = rfr.predict(test_features)
# Create a scatter plot with train and test actual vs predictions
plt.scatter(train_targets, train_predictions, label='train', alpha=0.5, color='b')
plt.scatter(test_targets, test_predictions, label='test', alpha=0.5, color='orangered')
plt.legend()
plt.show()
