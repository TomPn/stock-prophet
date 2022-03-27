# Stock Returns Prediction
A website that predicts whether a ticker will have positive returns in three months using machine learning, with additional stock information and price change plots presented. 

# Demo
Try it here.
screen shots

# Algorithm

There are three components in the process of building the website, each using different approachs and algorithms.

### Build the Dataset
The dataset contains two parts: features and target. The features includes volumn, dividend, current price, fiify day average, the difference between current price and fifty day average, and the 90-day returns. The target is a list of 0 and 1 where 1 indicates the ticker will have a positive return in three months and 0 indicates a negative return. The building of this dataset uses Yahoo Finance and Yahoo_fin throughoutly. 

The first step is to get a list of US-listed tickers, including S&P500, DowJones, NASDAQ stocks and more. Next, we collect raw data such as volume, fifty day average, dividend yield and closing prices on different dates. After that, we process the raw data to get features and the target(e.g. using closing prices on start date and mid date to calculate the 90-day return before the current date).Lastly, we output the dataset as a csv file. 

##### A Note on start_date, mid_date and end_date
The goal of this project is to predict the stock returns in three months, but in reality, it is impossible to look into the future. So the solution is to move the timeline three months backwards and set the 'current date' to be 90 days before, which is called mid_date. The start date is 90 days before the mid date, and is used to calculate the 90-day returns prior to the 'current date'. Similarly, the end date is 90 dys after the mid date, and is used to calculate the 90-day returns after the 'current date', which is the target that we are trying to predict. 


### Design SVM Model
The building of our model uses scikit-learn throughoutly.

Once the dataset is ready, the first thing to do is to normalize the feature values using min-max normalization, since the range of each feature is quite different. Then we separate the features and the target and split them into train sets and test sets. What after is the grid search, where the GridSearchCV function iterates each comnbination of parameters and returns the optimal parameters that produce the model with highest accuracy. Next, we repeatly retrain the model until the accuracy reaches 87% and lastly store the model and its accuracy using pickle.


### Display additional stock info
In addition to the predicted return, the user might want to know some additional information about the stock. 
The first step to do so is to fetch important stock information about the stock from Yahoo Finance and Yahoo_fin, which includes the open price, high and low of the day, dividend yield, p/e ratio, market capitalization, and 52 week high and low. In addition, the  1D and 1Y closing price are plotted using Matplotlib. For the 1D closing prices, We first determine the last day that stock market opens(they closes on weekends and certain holidays) using a series of if statements, then extract and plot the closing prices for the date with a 5-minute interval. For the 1Y closing prices, we simply extract and plot the closing prices from 365 days ago and today. 

