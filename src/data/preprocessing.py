import pandas as pd 
import numpy as np

df=pd.read_csv("C:\\Users\\a\\OneDrive\\Desktop\\Predictive-Analytics-for-Customer-Churn\\notebooks\\EDA_data.csv")

# convert all colimns in lower case 
df.columns=df.columns.str.lower()


# drop totalchess columns hase high coralated month cahnge also drop accountage i  creat new accounted age with agegroup base  
df.drop(columns=["totalcharges","accountage","customerid"],inplace=True)



# dive data X feture and y target 
X=df.drop(columns=["churn"])
y=df["churn"]


# split data train and test 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)


# to save splite data 
import os
 
os.makedirs("data/processed",exist_ok=True)
X_train.to_csv("data/processed/X_train.csv",index=False)
X_test.to_csv("data/processed/X_test.csv",index=False)


y_train.to_csv("data/processed/y_train.csv",index=False)
y_test.to_csv("data/processed/y_test.csv",index=False)



print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)