import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore


df = pd.read_csv(r"C:\Users\Sures\Downloads\Data Scientist\18th-- Cross validation\18th-- Cross validation\LOAN APPROVAL PREDICTION\test_case\Loan_Data.csv")

df.drop("Loan_ID",axis=1,inplace=True)


df["Credit_History"] = df["Credit_History"].replace({1:"Good",2:"Bad"})

categorical_features = ["Gender","Married","Dependents","Education","Self_Employed","Loan_Amount_Term","Credit_History","Property_Area","Loan_Status"]
numerical_features = ["ApplicantIncome","CoapplicantIncome","LoanAmount"]

for feature in categorical_features:
    df[feature] = df[feature].fillna(df[feature].mode()[0])
 
for feature in numerical_features:
    df[feature] = df[feature].fillna(df[feature].mean())



z_scores = df[numerical_features].apply(zscore)

outlier_filter = (np.abs(z_scores) <= 3).all(axis=1)

df = df[outlier_filter]
    

label = LabelEncoder()
df[categorical_features] = df[categorical_features].apply(label.fit_transform)



for i,feature in enumerate(numerical_features):
    plt.figure(i)
    plt.boxplot(df[feature])
    plt.show()
