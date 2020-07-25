import sklearn as sk
import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import math
df = pd.read_excel('RainData.xlsx')
df.drop(['Total'], 1, inplace=True)
df1 = pd.read_excel('FloodDates.xlsx')
df.set_index('Year', inplace=True)

month_year = []
rainfall = []
for i in df.index:
    rainfall.append(list(df.loc[i]))
    for j in df.loc[df.index==i]:
        month_year.append(j + f'{i}')

rainfall = np.array(rainfall).flatten()
month_year = np.array(month_year)
data = pd.DataFrame({'MonthYear': month_year, 'Rainfall': rainfall})

df1.sort_values(by='Year', inplace=True)
flooddata = []

temp1 = list(df1['Month'])
temp2 = list(df1['Year'])
for i in range(len(temp2)):
    flooddata.append(temp1[i] + f'{temp2[i]}')


output = np.nan
data['output'] = output

c = 0

for i in range(len(data['MonthYear'])):
    if c < len(flooddata):
        if data.at[i, 'MonthYear'] == flooddata[c]:
            data.at[i, 'output'] = 1
            c += 1
        else:
            data.at[i, 'output'] = 0
    else: 
        data = data.fillna(0)
        break


X = np.array(data.drop(['output', 'MonthYear'], 1))
# X = preprocessing.scale(X1)
y = np.array(data['output'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = LogisticRegression()
clf.fit(X_train, y_train)

acc = clf.score(X_test, y_test)
coeff = clf.coef_
inter = clf.intercept_
print(acc)
# pred = clf.predict_proba([[309]])

###########Shifted the unbiased sigmoid to the left to account for a High Positive Bias###########

def sigmoid(val):
    return 1/(1 + math.exp(-(coeff*val + inter/4)))

pred = clf.predict(X_test)
score = 0
for i in range(len(y_test)):
    if y_test[i] == pred[i]:
        score += 1

print('Enter mm of rainfall: ')
val = int(input())

acc = score/len(y_test)
print('Prediction: ', round(sigmoid(val)))
print('Prediction confidence: ', sigmoid(val)*100, '%')


