import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
iris_df=pd.read_csv('https://raw.githubusercontent.com/safal/DS- ML/main/Iris.csv')
print(iris_df.head())
x=iris_df.drop('Species',axis=1)
y=iris_df['Species']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratif
y=y)
classifier=GaussianNB()
model=classifier.fit(x_train,y_train)
y_train_pred=model.predict(x_train)
y_test_pred=model.predict(x_test)
print(accuracy_score(y_test,y_test_pred))
print(confusion_matrix(y_test,y_test_pred))
print(classification_report(y_test,y_test_pred))
