# The Iris dataset is referred to as a “toy dataset” because it has only 150 samples and four features. 
# The dataset describes 50 samples for each of three Iris flower species—Iris setosa, Iris versicolor and Iris 
# virginica. Each sample’s features are the sepal length, sepal width, petal 
# length and petal width, all measured in centimeters. The sepals are the larger outer parts of each flower 
# that protect the smaller inside petals before the flower buds bloom.

#EXERCISE
# load the iris dataset and use classification
# to see if the expected and predicted species
# match up

# display the shape of the data, target and target_names

# display the first 10 predicted and expected results using
# the species names not the number (using target_names)

# display the values that the model got wrong

# visualize the data using the confusion matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

print(iris.data.shape)
print(iris.target.shape)  
print(iris.target_names) #names of the classes, return as numpy array


import matplotlib.pyplot as plt 

figures,axes = plt.subplots(nrows=4, ncols=6, figsize=(6,4)) #6 by 4 grid

#plt.show() #to graph, plan to zip through each image, axis, image and target zip through them

#iterates 24 times, takes the least ones number 24
'''for item in zip(axes.ravel(), digits.images, digits.target):
    axes, image, target = item
    axes.imshow(image,cmap=plt.cm.gray_r) #imshow put image in box grayscale
    axes.set_xticks([]) #remove ticmarks empty list
    axes.set_yticks([])
    axes.set_title(target)
plt.tight_layout()
'''
#plt.show()

from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    iris.data, iris.target, random_state=11
    )

#same as ml1
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X=data_train, y=target_train) #data for the training mode and then the target for the training mode

#result would be array of numbers between 0-9 because it is given prediction of data results
predicted = knn.predict(X=data_test) #only have to give it X because we are trying to predict the target
expected = target_test #compare predicted to expected

print(predicted[:20])
print(expected[:20]) #one digit second to last is wrong, not perfect, pretty close
#results: the numbers tell us the class number of flowers, but we want the target names as the output so

predicted = [iris.target_names[x] for x in predicted]
expected = [iris.target_names[y] for y in expected]

#print(predicted[:20])
#print(expected[:20])

wrong = [(p,e) for (p,e) in zip(predicted, expected) if p != e] #put combination in output if not equal to each other

print(wrong)

from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(y_true = expected, y_pred = predicted)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt2

confusion_df = pd.DataFrame(confusion, index=iris.target_names, columns=iris.target_names)

figure = plt2.figure(figsize=(7,6))
axes = sns.heatmap(confusion_df, annot=True, cmap=plt2.cm.nipy_spectral_r)
plt.xlabel("Expected")
plt.ylabel("Predicted")
plt.show()

print('done')
