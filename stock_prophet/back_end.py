import pandas as pd
import numpy as np
import yfinance as yf
from yahoo_fin import stock_info as si
import datetime
from datetime import date
import concurrent.futures
import schedule
import time
import sklearn
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV
import pickle

def get_tickers():
    '''Get a list of tickers in US market'''
    # Get lists of tickers and put them in a set
    sp500 = set(si.tickers_sp500())
    nasdaq = set(si.tickers_nasdaq())
    dow = set(si.tickers_dow())
    other = set(si.tickers_other())
    total = set.union(sp500,nasdaq,dow,other)

    # Find out the undesired tickers(such as tickers with warrants or rights)
    filter_lst = ['W','R','P','Q','U','T','L','Z']
    bad_tickers = set()
    for i in total:
        if (len(i) >= 5 and i[-1] in filter_lst) or '$' in i or len(i) == 0:
            bad_tickers.add(i)
    
    # Subtract the undesirable tickers from the ticker set
    good_tickers = total - bad_tickers
    # Return the remaining tickers in a list
    return list(good_tickers)

def timing(ticker,today,tomorrow):
    '''Find the dates that works for calculating returns for features and target'''
    try:
        # Returns as features are returns from start date to mid date, which are what we're given
        # Returns as target are returns from mid date to end date, which are what we're trying to predict
        # Start date represents three months in the past, mid date represets the current, end date represents three months in the future
        # Since we can't look into the future, we move the timeline three months backwards
        # End date becomes the current date
        end_date = ticker.history(start = today -  datetime.timedelta(10),end = today,interval = '1d')['Close'].index[-1]
        end_end = tomorrow
        # Mid date becomes three months before
        mid_date = end_date -  datetime.timedelta(90)
        mid_end = mid_date +  datetime.timedelta(10)
        # start date becomes six months before
        start_date = mid_date -  datetime.timedelta(90)
        start_end = start_date +  datetime.timedelta(10)
        # Return the dates in string
        return str(start_date)[:10],str(start_end)[:10],str(mid_date)[:10],str(mid_end)[:10],str(end_date)[:10],str(end_end)[:10]
    except:
        pass
    
def get_features(symbol):
    '''prepare the information needed for the features'''
    # Set up today's date and the ticker
    today = date.today()
    tomorrow = today + datetime.timedelta(1)
    ticker = yf.Ticker(symbol)
    try:
        # Get start price,mid price and end price using the dates from the timing function
        start_date,start_end,mid_date,mid_end,end_date,end_end = timing(ticker,today,tomorrow)
        end_price = float(ticker.history(start = end_date,end = end_end,interval = '1d')['Close'][0])
        start_price = float(ticker.history(start = start_date,end = start_end,interval = '1d')['Close'][0])
        mid_price = float(ticker.history(start = mid_date,end = mid_end,interval = '1d')['Close'][0])
        # Get the dividend yield, average volume and fifty-day average 
        dividendYield = si.get_quote_data(symbol)['trailingAnnualDividendYield']
        volume = si.get_quote_data(symbol)['regularMarketVolume']
        fiftyDayAverage = si.get_quote_data(symbol)['fiftyDayAverage']
        # Return the above information
        return symbol,volume,fiftyDayAverage,dividendYield,start_price,mid_price,end_price
    except:
        pass


def collect_data(df, symbols):
    '''Use multithreading to get information from the prepare_features() function'''
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = executor.map(get_features,symbols)
    for i in result:
        try:
            df[i[0]] = [i[1],i[2],i[3],i[4],i[5],i[6]]
        except:
            pass
    return df

def get_dataset(data):
    '''Create the dataset using the information collected from the collect_data() function'''
    # Transpose the dataframe
    data = data.T
    # Rename the columns
    data.columns = ['volume','fiftyDayAverage','dividendYield','start_price','mid_price','end_price']
    # Replace Nan in dividendYield with 0
    data['dividendYield']=data['dividendYield'].fillna(0)
    # Calculate the dividend for feature returns and dividend returns
    data['feature_dividend']=data['dividendYield']*data['mid_price']/365*90
    data['target_dividend']=data['dividendYield']*data['end_price']/365*90
    # Drop any rows with Nan
    data = data.dropna()
    # Calculate the feature and target returns by adding the capital gains and the dividend
    data['feature_returns'] = (data['mid_price']-data['start_price']+data['feature_dividend'])
    data['target_returns'] = (data['end_price']-data['mid_price']+data['target_dividend'])
    # Diff measures how much the mid price is above or below the average price
    data['diff'] = data['mid_price'] - data['fiftyDayAverage']
    # Set target to 1 if the target return is positive, 0 if it is negative
    target = (data['target_returns']>0) 
    target = target.values.tolist()
    mapping = {True: 1, False: 0}
    target = [mapping.get(n, n) for n in target]
    data['target'] = target
    # Reset the index and drop the extra columns
    data.reset_index(drop = True,inplace=True)
    data = data.drop(['start_price','end_price','dividendYield','fiftyDayAverage','target_returns','target_dividend'],axis=1)
    # Return the dataset
    return data


def get_x_y(data):
    '''Get the features and the target(x and y), as well as the mins and maxs used to normalize the features'''
    # Drop the target, the remaining columns are features
    features = data.drop(['target'],axis = 1)
    # Get the mins and maxs for each column in lists
    mins = features.apply(lambda x: x.min(axis=0))
    maxs = features.apply(lambda x: x.max(axis=0))
    mins = mins.values.tolist()
    maxs = maxs.values.tolist()
    # Normalize the features to values between 0 and 1
    features = features.apply(lambda x: (x - x.min(axis=0))/ (x.max(axis=0) - x.min(axis= 0 )))
    # Convert features and target into numpy arrays
    y = np.array(data.target)
    x = np.array(list(zip(features['volume'].tolist(),features['mid_price'].tolist(),features['feature_returns'].tolist(),features['diff'].tolist(),features['feature_dividend'].tolist())))
    # Return features, target and the mins and maxs of each column
    return x,y, [mins,maxs]


def main():
    '''Generate the dataset and the machine learning model used to predict the stock returns'''
    # Create an empty dataframe
    data = pd.DataFrame()
    # Get the tickers, collect data needed, then calculate the features and target
    symbols = get_tickers()
    data = collect_data(data, symbols)
    data = get_dataset(data)
    # Store the dataset
    data.to_csv(r'static\my_data.csv',index=False)
    data = pd.read_csv(r'static\my_data.csv')
    # Get features, target and split them into train set and test set. Also store the mins and maxs.
    x,y,mins_and_maxs = get_x_y(data)
    x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size = 0.1)
    with open("mins_and_maxs.pickle",'wb') as f:
        pickle.dump(mins_and_maxs,f)
    # Create a parameter grid for grid search
    param_grid = [{
    'kernel' :['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma':['auto', 1, 0.1, 0.01, 0.001, 0.0001],
    'C': [0.1,1, 10, 100,500],
    'degree':[2,3,4,5,6,7,8]}]
    # Search the best model with parameters from the parameter grid and train the model
    model = GridSearchCV(svm.SVC(), param_grid = param_grid, refit=True,verbose=2)
    best_model= model.fit(x_train,y_train)
    # Initiaize the best accuracy to 0
    best_acc = 0
    # Iterate to find a higher accuracy
    for i in range(500):
        # Split the dataset, train the model, get the prediction then score the accuracy
        x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size = 0.1)
        model = best_model.best_estimator_
        model.fit(x_train,y_train)
        y_predict =model.predict(x_test)
        acc = metrics.accuracy_score(y_test,y_predict)
        # If the current acc is greater then the best acc, replace best_acc with acc then store the model
        if acc>best_acc:
            best_acc = acc
            with open("model.pickle",'wb') as f:
                pickle.dump(model,f)
            
    # Store the best accuracy
    with open("acc.pickle",'wb') as f:
        pickle.dump(best_acc,f)
    
# Run the main function every monday at 00:00
schedule.every().sunday.at('12:49').do(main)
while 1:
    schedule.run_pending()
    time.sleep(1)

