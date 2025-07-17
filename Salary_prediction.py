
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# sc=StandardScaler()
df = pd.read_csv('Salary_prediction_data.csv')
le=LabelEncoder()
df['Internship']=le.fit_transform(df['Internship'])
df['Hackathon']=le.fit_transform(df['Hackathon'])
df['PlacementStatus']=le.fit_transform(df['PlacementStatus'])
df.drop(columns=['Unnamed: 0','StudentId'],inplace=True)

x=df.drop(columns=['salary'])
y=df['salary']
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=2)
# xtrain=sc.fit_transform(xtrain)
# xtest=sc.transform(xtest)
rf_reg = RandomForestRegressor(random_state=2)
rf_reg.fit(xtrain, ytrain)
y_pred_rf = rf_reg.predict(xtest)
print("Random Forest Regressor R2 score:", r2_score(ytest, y_pred_rf))
print("Random Forest Regressor MSE:", mean_squared_error(ytest, y_pred_rf))
pickle.dump(rf_reg, open('model1.pkl','wb'))
model1 = pickle.load(open('model1.pkl','rb'))
sample=pd.DataFrame([[6.7,	0	,1,	1	,7	,15,	0,	1,	55,	72,	2,	1]], columns=x.columns) 
# sample_arr = sc.transform(sample)     # scale with same scaler
print(rf_reg.predict(sample))
