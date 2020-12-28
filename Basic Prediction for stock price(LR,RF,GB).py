import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import talib as ta
from sklearn import preprocessing
from sklearn import linear_model as lm
from sklearn import tree
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV


data = pd.read_csv('TSLA.csv')
data['Date'] = pd.to_datetime(data['Date'], format='%Y/%m/%d')
data.set_index('Date', inplace=True)
data['5d_future_close'] = data['Adj_Close'].shift(-5)
data['5d_close_future_pct'] = data['5d_future_close'].pct_change(5)
data['5d_close_pct'] = data['Adj_Close'].pct_change(5)
data['Volume_1d_change'] = data['Volume'].pct_change(1)
feature_names = ['5d_close_pct', 'Volume_1d_change']
for n in [14, 30, 100]:
    data['sma' + str(n)] = ta.SMA(data['Adj_Close'].values, timeperiod=n)
    data['rsi' + str(n)] = ta.RSI(data['Adj_Close'].values, timeperiod=n)
    data['ema' + str(n)] = ta.EMA(data['Adj_Close'].values, timeperiod=n)
    data['std' + str(n)] = ta.STDDEV(data['Adj_Close'].values, timeperiod=n)
    feature_names = feature_names + ['sma' + str(n), 'rsi' + str(n), 'ema' + str(n), 'std' + str(n)]
data['ATR'] = ta.ATR(data['High'].values,
                     data['Low'].values,
                     data['Adj_Close'].values,
                     timeperiod=14)
data['ADX'] = ta.ADX(data['High'].values,
                     data['Low'].values,
                     data['Adj_Close'].values,
                     timeperiod=14)
macd, macdsignal, macdhist = ta.MACD(data['Adj_Close'].values,
                                     fastperiod=12,
                                     slowperiod=26,
                                     signalperiod=9)
data['MACD'] = macd
data['MACDsignal'] = macdsignal
feature_names = feature_names + ['ATR', 'ADX', 'MACD', 'MACDsignal']
data.tail()
data = data.dropna()

################################################################
features = data[feature_names]
targets = pd.DataFrame(data['Adj_Close'])
F_scaled = preprocessing.MinMaxScaler().fit_transform(features)
features = pd.DataFrame(F_scaled)
features.columns = feature_names
T_scaled = preprocessing.MinMaxScaler().fit_transform(targets)
#targets = pd.DataFrame(T_scaled)
ttt = pd.DataFrame(T_scaled)
ttt.columns = ['Adj_Close']
targets = T_scaled.ravel()  #Change it to 1D array
total = features.join(ttt)
#scatter_matrix(total)


train_size = int(0.8 * targets.shape[0])
train_features = features[:train_size]
train_targets = targets[:train_size]
test_features = features[train_size:]
test_targets = targets[train_size:]
print('Training Features Shape:', train_features.shape)
print('Training Targets Shape:', train_targets.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Targets Shape:', test_targets.shape, '\n')

skmodel = lm.LinearRegression().fit(train_features,train_targets)
def LM(skmodel, train_features, train_targets ,test_features, test_targets):
    print('LM train', skmodel.score(train_features, train_targets))
    print('LM test', skmodel.score(test_features, test_targets))
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
LM(skmodel,train_features, train_targets, test_features, test_targets)


def DTR(train_features, train_targets, test_features, test_targets):
    score =[]
    for i in range(1, 10):
        decision_tree1 = DecisionTreeRegressor(max_depth=i)
        decision_tree1.fit(train_features, train_targets)
        score.append(decision_tree1.score(test_features, test_targets))
    print('best depth', np.argmax(score)+1)
    decision_tree = DecisionTreeRegressor(max_depth=np.argmax(score)+1)
    decision_tree.fit(train_features, train_targets)
    print('DTR train', decision_tree.score(train_features, train_targets))
    print('DTR test', decision_tree.score(test_features, test_targets), '\n')

    plt.figure(figsize=(20, 15))
    tree.plot_tree(decision_tree, filled=True)
    plt.show()
    train_predictions = decision_tree.predict(train_features)
    test_predictions = decision_tree.predict(test_features)
    # Scatter the predictions vs actual values
    plt.scatter(train_predictions, train_targets, label='train')
    plt.scatter(test_predictions, test_targets, label='test')
    plt.legend()
    plt.show()
DTR(train_features, train_targets, test_features, test_targets)


def RF(train_features, train_targets, test_features, test_targets):
    rfr = RandomForestRegressor(n_estimators=400,
                                random_state=42)
    rfr.fit(train_features, train_targets)
    # Look at the R^2 scores on train and test
    print('RF train', rfr.score(train_features, train_targets))
    print('RF test',rfr.score(test_features, test_targets), '\n')
    train_predictions = rfr.predict(train_features)
    test_predictions = rfr.predict(test_features)
    plt.scatter(train_targets, train_predictions, label='train')
    plt.scatter(test_targets, test_predictions, label='test')
    plt.legend()
    plt.show()
RF(train_features, train_targets, test_features, test_targets)




def GB(train_features, train_targets, test_features, test_targets):
    gbr = GradientBoostingRegressor(n_estimators=400,
                                    random_state=42)
    gbr.fit(train_features, train_targets)
    print('GB train', gbr.score(train_features, train_targets))
    print('GB test', gbr.score(test_features, test_targets), '\n')
    train_predictions = gbr.predict(train_features)
    test_predictions = gbr.predict(test_features)
    plt.scatter(train_targets, train_predictions, label='train')
    plt.scatter(test_targets, test_predictions, label='test')
    plt.legend()
    plt.show()
GB(train_features, train_targets, test_features, test_targets)
