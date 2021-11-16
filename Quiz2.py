'''These files contain data for video game ratings given out by the ESRB. 
Both files contain 34 features.
32 of these contribute to the ESRB's assigned rating. 
The data has already been split into a train and 
a test file for you to use to train and test your model. 
Your goal is to create a model that will take the 32 features
for each game and predict a rating. Keep in mind that the final output 
should be the name of the target rating and not the number. 
Please refer to this file which should match your output.

Using methods covered in class:

1) load the dataset and use the KNeighborsClassifier to train and test your model

2) Display all wrong predicted and expected pairs

3) produce a csv file of the name of the game and the predicted rating

Submit your completed python file.'''


import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import csv

game_ratings = pd.read_csv("gameratings.csv")
test = pd.read_csv("test_esrb.csv")
names = pd.read_csv("target_names.csv")
x = game_ratings[
    [

        "alcohol_reference",

        "animated_blood",

        "blood",

        "blood_and_gore",

        "cartoon_violence",

        "crude_humor",

        "drug_reference",

        "fantasy_violence",

        "intense_violence",

        "language",

        "lyrics",

        "mature_humor",

        "mild_blood",

        "mild_cartoon_violence",

        "mild_fantasy_violence",

        "mild_language",

        "mild_lyrics",

        "mild_suggestive_themes",

        "mild_violence",

        "no_descriptors",

        "nudity",

        "partial_nudity",

        "sexual_content",

        "sexual_themes",

        "simulated_gambling",

        "strong_janguage",

        "strong_sexual_content",

        "suggestive_themes",

        "use_of_alcohol",

        "use_of_drugs_and_alcohol",

        "violence",
    ]
].values


y = game_ratings[["Target"]].values.reshape(-1, 1)
test_train = test[

    [
        "alcohol_reference",

        "animated_blood",

        "blood",

        "blood_and_gore",

        "cartoon_violence",

        "crude_humor",

        "drug_reference",

        "fantasy_violence",

        "intense_violence",

        "language",

        "lyrics",

        "mature_humor",

        "mild_blood",

        "mild_cartoon_violence",

        "mild_fantasy_violence",

        "mild_language",

        "mild_lyrics",

        "mild_suggestive_themes",

        "mild_violence",

        "no_descriptors",

        "nudity",

        "partial_nudity",

        "sexual_content",

        "sexual_themes",

        "simulated_gambling",

        "strong_janguage",

        "strong_sexual_content",

        "suggestive_themes",

        "use_of_alcohol",

        "use_of_drugs_and_alcohol",

        "violence",
    ]

].values


test_target = test[["Target"]].values.ravel()
games = test[["title"]].values.ravel()
print(x.shape)
print(y.shape)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X=x, y=y.ravel())
predicted = knn.predict(X=test_train)
expected = test_target

a = []
b = []
Dict = {}

with open("target_names.csv", mode="r") as tn:
    reader = csv.reader(tn)
    csv_dict = {rows[0]: rows[2] for rows in reader}

print('                                                                  ')

print(csv_dict)
print('                                                                  ')
print(csv_dict.get('1'))

for n in predicted:
    a.append(csv_dict.get(str(n)))

for n in expected:
    b.append(csv_dict.get(str(n)))

results = []

for n in zip(games,a,b):
    results.append(n)
print('                                                                  ')

print('The percentages of Right guesses is:' )

print(format(knn.score(test_train, test_target), ".2%"))

wrong = [(game_l,pred,exp) for (game_l,pred,exp) in zip(games,a,b) if pred != exp]

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)

pd.set_option('display.width', None)

pd.set_option('display.max_colwidth', None)

wrong = pd.DataFrame(wrong, columns=['title','predicted','expected'])
print('                                                                  ')

print('Wrong guesses are:')
print(wrong)

header = ['title','predicted','expected']
with open('games_result.csv','w',newline='') as file:
    obj= csv.writer(file)
    obj.writerow(header)
    obj.writerows(results)
