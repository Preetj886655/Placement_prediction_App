import pandas as pd
import pickle
import numpy as np
df = pd.read_csv(r'Placement_Prediction_data.csv')
df.sample(5)

df.info()

df.shape

df.isnull().sum()

df.duplicated().sum()

#now we are going to drop the column unnamed 0 and studentId
df.drop(columns=['Unnamed: 0','StudentId'],inplace=True)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#now we to encode the intership and hackathon
le=LabelEncoder()
df['Internship']=le.fit_transform(df['Internship'])
df['Hackathon']=le.fit_transform(df['Hackathon'])


print(df.sample(5))

# from sklearn.preprocessing import StandardScaler
# sc=StandardScaler()


df.describe()

df.sample(5)

#now we are going to train test split
x=df.drop(columns=['PlacementStatus'])
y=df['PlacementStatus']
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=2)
# xtrain=sc.fit_transform(xtrain)
# xtest=sc.transform(xtest)
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(xtrain,ytrain)
pred=rf.predict(xtest)
print(accuracy_score(ytest,pred))

pred=rf.predict(xtrain)
print(accuracy_score(ytrain,pred))
pickle.dump(rf, open('model.pkl','wb'))
model1 = pickle.load(open('model.pkl','rb'))
sample=pd.DataFrame([[8.9,0,3,2,9,4.0,1,1,78,82,0]], columns=x.columns) 
# sample_arr = sc.transform(sample)     # scale with same scaler
print(rf.predict(sample))







