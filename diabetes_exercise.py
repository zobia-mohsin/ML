''' Using the Diabetes dataset that is in scikit-learn, answer the questions below and create a scatterplot
graph with a regression line '''

import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets

diabetes = datasets.load_diabetes()
#how many sameples and How many features?
print(diabetes.data.shape)

# What does feature s6 represent?
print(diabetes.DESCR)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, random_state=11) #split 75 25

#print out the coefficient

mymodel = LinearRegression()

mymodel.fit(X_train, y_train) #mymodel.fit function variables that we called it

print(mymodel.coef_)
print(mymodel.intercept_)
#print out the intercept
predicted = mymodel.predict(X_test)
expected = y_test
# create a scatterplot with regression line
#expected vs predicted which is the y value
plt.plot(expected, predicted, '.')
x = np.linspace(0,330,100) #an array of x values 100 values between 0 and 330
y = x
plt.plot(x,y)
plt.show()


#QUIZ, rating either 1,2,3,4 which is the target of the game. based on 32 feautes, the target rating is produced.
#target_names file target name based on number, first file for training, second file is testing file.
#Name of game and rating for test file, rating should be in words (everyone, etc). similar to ex 1, no confusion matrix
#or no linear regression 