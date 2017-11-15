'''
    This is a first playground for XGboost, an implementation of gradient boosting
    Which is a subset of ensemble learning.
    Here we are experimenting with a binary classifier.. 
    A relatively simple task. but just for the sake of using XGboost
'''

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('pima-indians-diabetes.data.txt')
df_array = df.values

X = df_array[:,:8]
y = df_array[:,8]

#Split training to test set
seed = 7
test_size = 0.33
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = test_size, random_state=seed)

model = XGBClassifier()
model.fit(X_train,y_train)

#Make predictions for the test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test,predictions)
print("Accuracy of this model is: ", accuracy)

