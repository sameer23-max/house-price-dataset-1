import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv('student_pass_dataset.csv')
le=LabelEncoder()
df['Has_Tutor']=le.fit_transform(df['Has_Tutor'])
df['Passed']=le.fit_transform(df['Passed'])
X=df[['Study_hours','Sleep_hours','Attendance_pct','Has_Tutor']]
Y=df['Passed']
sc=StandardScaler()
scaled=pd.DataFrame(sc.fit_transform(df[['Study_hours','Sleep_hours','Attendance_pct']]))


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=7)
model=LogisticRegression()
model.fit(x_train,y_train)
ans=model.predict(x_test)

print(model.score(x_test,y_test))
