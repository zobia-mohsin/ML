import pandas as pd

nyc = pd.read_csv("ave_hi_nyc_jan_1895-2018.csv")

print(nyc.head(3)) #prints first three lines, every column in a dataframe is a series. 
#A series only has one column so convert it into 2 dimensional.
print(nyc.Date.values) #result in one dimensional array, HAVE TO CONVERT TO 2 dimensional

print(nyc.Date.values.reshape(-1,1)) #infer the number of rows based on number of cloumns (-1)

print(nyc.Temperature.values)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(nyc.Date.values.reshape(-1,1), nyc.Temperature.values, random_state=11) #split 75 25

#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_test.shape)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X=X_train, y=y_train)

coef = lr.coef_
intercept = lr.intercept_

predicted = lr.predict(X_test)
expected = y_test

print(predicted[:10])
print(expected[:10])
#results not at all accurate
predict = lambda x: coef * x + intercept #mx+b

print(predict(2025))

import seaborn as sns 
axes = sns.scatterplot(data = nyc,x="Date",y='Temperature', palette= 'winter', legend=False)

axes.set_ylim(10,70)

import numpy as np

x= np.array([min(nyc.Date.values), max(nyc.Date.values)])