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
pd.set_option('display.max_columns', None)
print(data)
data['Date'] = pd.to_datetime(data['Date'], format='%Y/%m/%d')
data.set_index('Date', inplace=True)
data['1d_future_close'] = data['Adj_Close'].shift(-1)
data['1d_close_pct'] = data['Adj_Close'].pct_change(1)
data['Volume_1d_change'] = data['Volume'].pct_change(1)
feature_names = ['1d_close_pct', 'Volume_1d_change']
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
data = data.dropna()
print(data.columns)
################################################################
print(data)
features = data[feature_names]
targets = pd.DataFrame(data['1d_future_close'])


train_size = int(0.8 * targets.shape[0])
train_features = features[:train_size]
train_features = pd.DataFrame(preprocessing.StandardScaler().fit_transform(train_features))
train_features.columns = feature_names
train_targets = targets[:train_size]
train_targets = train_targets.values.ravel()

test_features = features[train_size:]
test_features = pd.DataFrame(preprocessing.StandardScaler().fit_transform(test_features))
test_features.columns = feature_names
test_targets = targets[train_size:]
test_targets = test_targets.values.ravel()


print('Training Features Shape:', train_features.shape)
print('Training Targets Shape:', train_targets.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Targets Shape:', test_targets.shape, '\n')


skmodel = lm.LinearRegression().fit(train_features,train_targets)
def LM(skmodel, train_features, train_targets ,test_features, test_targets):
    print('LR train', skmodel.score(train_features, train_targets))
    print('LR test', skmodel.score(test_features, test_targets))
    print('intercept = ', skmodel.intercept_, '\n', 'slope=', skmodel.coef_, '\n')
    train_predictions = skmodel.predict(train_features)
    test_predictions = skmodel.predict(test_features)
    plt.scatter(train_targets, train_predictions, label='train', alpha=0.4, color='b')
    plt.scatter(test_targets, test_predictions, label='test', alpha=0.4, color='r')
    #plt.scatter(targets, targets, label='Original', alpha=0.3, color='navajowhite')
    # Plot the perfect prediction line
    xmin, xmax = plt.xlim()
    plt.plot(np.arange(xmin, xmax, 0.01), np.arange(xmin, xmax, 0.01), c='k')
    plt.xlabel('actual')
    plt.ylabel('predictions')
    plt.title('Linear Regression')
    plt.legend()
    plt.show()
LM(skmodel,train_features, train_targets, test_features, test_targets)


def DTR(train_features, train_targets, test_features, test_targets):
    score =[]
    for i in range(1, 10):
        decision_tree1 = DecisionTreeRegressor(max_depth=i)
        decision_tree1.fit(train_features, train_targets)
        score.append(decision_tree1.score(test_features, test_targets))
    print('DTR best depth', np.argmax(score)+1)
    decision_tree = DecisionTreeRegressor(max_depth=np.argmax(score)+1)
    decision_tree.fit(train_features, train_targets)
    print('DTR train', decision_tree.score(train_features, train_targets))
    print('DTR test', decision_tree.score(test_features, test_targets), '\n')

    plt.figure(figsize=(35, 25))
    tree.plot_tree(decision_tree, filled=True, rounded=True, feature_names=feature_names)
    plt.show()
    train_predictions = decision_tree.predict(train_features)
    test_predictions = decision_tree.predict(test_features)
    # Scatter the predictions vs actual values
    plt.scatter(train_predictions, train_targets, label='train', alpha = 0.6, c='b')
    plt.scatter(test_predictions, test_targets, label='test', alpha = 0.6, c='r')
    plt.xlabel('actual')
    plt.ylabel('predictions')
    plt.title('Decision Tree')
    plt.legend()
    plt.show()
DTR(train_features, train_targets, test_features, test_targets)


def RF(train_features, train_targets, test_features, test_targets):
    score = []
    for i in range(1, 10):
        decision_tree1 = RandomForestRegressor(max_depth=i)
        decision_tree1.fit(train_features, train_targets)
        score.append(decision_tree1.score(test_features, test_targets))
    print('RF best depth', np.argmax(score)+1)
    rfr = RandomForestRegressor(n_estimators=400,
                                max_depth=np.argmax(score)+1,
                                random_state=42)
    rfr.fit(train_features, train_targets)
    # Look at the R^2 scores on train and test
    print('RF train', rfr.score(train_features, train_targets))
    print('RF test',rfr.score(test_features, test_targets), '\n')
    train_predictions = rfr.predict(train_features)
    test_predictions = rfr.predict(test_features)
    plt.scatter(train_targets, train_predictions, label='train', alpha = 0.6, c='b')
    plt.scatter(test_targets, test_predictions, label='test', alpha = 0.6, c='r')
    plt.xlabel('actual')
    plt.ylabel('predictions')
    plt.title('Random Forest')
    plt.legend()
    plt.show()
RF(train_features, train_targets, test_features, test_targets)


def GB(train_features, train_targets, test_features, test_targets):
    score = []
    for i in range(1, 10):
        score = []
        decision_tree1 = GradientBoostingRegressor(max_depth=i)
        decision_tree1.fit(train_features, train_targets)
        score.append(decision_tree1.score(test_features, test_targets))
    print('GBR best depth', np.argmax(score)+1)
    gbr = GradientBoostingRegressor(n_estimators=400,
                                    random_state=42,
                                    max_depth=np.argmax(score)+1
                                    )
    gbr.fit(train_features, train_targets)
    print('GBR train', gbr.score(train_features, train_targets))
    print('GBR test', gbr.score(test_features, test_targets), '\n')
    train_predictions = gbr.predict(train_features)
    test_predictions = gbr.predict(test_features)
    plt.scatter(train_targets, train_predictions, label='train', alpha = 0.6, c='b')
    plt.scatter(test_targets, test_predictions, label='test', alpha = 0.6, c='r')
    plt.xlabel('actual')
    plt.ylabel('predictions')
    plt.title('Gradient Boosting')
    plt.legend()
    plt.show()
GB(train_features, train_targets, test_features, test_targets)
