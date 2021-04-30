#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model, datasets
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

name = ['Area', 'Perimeter', 'Compactness','Length of kernel', 'Width of kernel','Asymmetry coefficient',
      'Length of kernel groove', 'Class (1, 2, 3)']

df = pandas.read_table('seeds_dataset.txt', delim_whitespace=True, names=name)

s1 = df[['Area','Perimeter']]
s2 = df[['Compactness', 'Length of kernel']]
s3 = df[['Width of kernel', 'Asymmetry coefficient']]
s4 = df[['Length of kernel groove', 'Class (1, 2, 3)']]

s1.plot.hist(alpha=0.75)
s2.plot.hist(alpha=0.75)
s3.plot.hist(alpha=0.75)
s4.plot.hist(alpha=0.75)


normalized = preprocessing.normalize(df)
print("Normalized Data = ", normalized)

X= df[name]
y = df['Class (1, 2, 3)']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_train, y_train)  # apply scaling on training data
y_pred=pipe.predict(X_test)
pipe.score(X_test, y_test)

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print("visualize the performance= ",cnf_matrix)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:




