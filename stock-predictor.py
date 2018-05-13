import glob
import datetime
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quandl
from matplotlib import style
from sklearn import cross_validation, preprocessing, svm
from sklearn.linear_model import LinearRegression


def get_stock_data(quandl_stock_name, forecast_time=5):
    try:
        df = quandl.get(quandl_stock_name)
    except SyntaxError:
        print("{} is an invalid database code format".format(quandl_stock_name))
    # * Removing unajusted data
    df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
    # * Adding
    df['HL_Perc'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
    df['HL_Perc_2'] = (df['Adj. High'] - df['Adj. Low']) / \
        df['Adj. Close'] * 100.0
    # * crude measure of volatility
    df['Perc_change'] = (df['Adj. Close'] - df['Adj. Open']
                         ) / df['Adj. Open'] * 100.0
    # # * filling missing values with -99999
    df.fillna(value=-99999, inplace=True)
    forecast_col = 'Adj. Close'
    forecast_out = forecast_time
    df['label'] = df[forecast_col].shift(-forecast_out)
    df.to_hdf('stockdata.hdf', quandl_stock_name, append=True)
    return df


def train_best_classifier(quandl_stock_name=None, df=None, forecast_time=5):
    if quandl_stock_name:
        try:
            df = pd.read_hdf("stockdata.hdf", quandl_stock_name)
        except KeyError:
            df = get_stock_data(quandl_stock_name)
    elif df:
        df = df
    else:
        print("ERROR: NO DATA PROVIDED.", file=sys.stderr)

    df.dropna(inplace=True)
    X = np.array(df.drop(['label'], 1))
    X = preprocessing.scale(X)
    y = np.array(df['label'])
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, test_size=0.2)

    estimators = [
        ["SVR_LINEAR",
         svm.SVR('linear'),
         -999
         ],
        ["SVR_POLY",
         svm.SVR('poly'),
         -999
         ],
        ["SVR_RBF",
         svm.SVR('rbf'),
         -999
         ],
        ["SVR_SIGMOID",
         svm.SVR('sigmoid'),
         -999
         ],
        ["LINREGRESSION",
         LinearRegression(n_jobs=-1),
         -999
         ],
    ]
    for i in estimators:
        clf = i[1]
        clf.fit
        clf.fit(X_train, y_train)
        confidence = clf.score(X_test, y_test)
        i[2] = confidence
        print("{} : {}".format(i[0], confidence))

    sorted_estimators = sorted(estimators, key=lambda x: x[2])

    save_stock_name = quandl_stock_name.replace("/", "_")
    p_fname = save_stock_name + "-" + sorted_estimators[-1][0]
    with open('{}.pickle'.format(p_fname), 'wb') as f:
        pickle.dump(sorted_estimators[-1][1], f)

    return sorted_estimators[-1][1]


def predictNplot(quandl_stock_name, forecast_time=5):

    save_stock_name = quandl_stock_name.replace("/", "_")
    stock_classifiers = glob.glob("./{}-*.pickle".format(save_stock_name))
    pickle_in = open(stock_classifiers[0], 'rb')
    clf = pickle.load(pickle_in)

    #  * Data
    df = pd.read_hdf("stockdata.hdf", quandl_stock_name)
    X = np.array(df.drop(['label'], 1))
    X = preprocessing.scale(X)
    X_lately = X[-forecast_time:]
    df['Forecast'] = np.nan

    forecast_set = clf.predict(X_lately)
    last_date = df.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 60 * 60 * 24
    next_unix = last_unix + one_day

    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += 86400
        df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

    style.use('ggplot')
    df['Adj. Close'].plot()
    df['Forecast'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()
    return forecast_set


def get_predictions(stock_name, forecast_time=5):

    # Get latest stock data, regarldess if it exists.
    get_stock_data(stock_name)

    files = glob.glob("./{}-*.pickle".format(stock_name.replace("/", "_")))
    if not files:
        train_best_classifier(stock_name)

    predictions = predictNplot(stock_name)

    return predictions


get_predictions("WIKI/WMT")
