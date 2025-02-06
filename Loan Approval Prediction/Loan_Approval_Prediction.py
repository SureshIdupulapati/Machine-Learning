import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv(r"Loan_Data.csv")
df.head()


df["Loan_Amount_Term"] = df["Loan_Amount_Term"].astype(object)
df["Credit_History"] = df["Credit_History"].astype(object)


df.drop("Loan_ID",axis=1,inplace=True)
df["Dependents"] = df["Dependents"].replace({"3+":"3"})
df["Education"] = df["Education"].replace({"Not Graduate":"Non Graduate"})



for feature in df.select_dtypes(include="object").columns:
     df[feature] =  df[feature].fillna( df[feature].mode()[0])
     
     
for feature in df.select_dtypes(include=["int","float"]).columns:
     df[feature] =  df[feature].fillna( df[feature].mean())
     

df["Income"] = df["ApplicantIncome"]+df["CoapplicantIncome"]
df.drop(["ApplicantIncome","CoapplicantIncome"],axis=1,inplace=True)


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
for feature in df.select_dtypes(include="object").columns:
     df[feature] =  label.fit_transform(df[feature]).astype(float)


df['Dependents'] = df['Dependents'].astype('int')
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].astype('int')


df['Loan_Amount_Term'] = df['Loan_Amount_Term']/12

X = df.iloc[:,[7,8,10]]
y = pd.DataFrame(df["Loan_Status"])


Train = []
Test = []
CV = []
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
for i in range(0,101):
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = i)
    model = LogisticRegression()
    model.fit(x_train,y_train)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    

    Train.append(accuracy_score(y_train,y_train_pred))
    Test.append(accuracy_score(y_test,y_test_pred))


    cv_scores = cross_val_score(model,x_train,y_train,cv=5,scoring = "accuracy")
    CV.append(cv_scores.mean())
em = pd.DataFrame({"Train":Train,"Test":Test,"CV":CV})
gm = em [(abs(em["Train"]-em["Test"])<=0.5) & (abs(em["Test"]-em["CV"])<=0.5)]
rs = gm[gm['CV']==gm['CV'].max()].index.to_list()[0]

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = rs)


'''
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(x_train,y_train)

ypred_train = log_model.predict(x_train)
ypred_test = log_model.predict(x_test)

print('Train Accuracy:',accuracy_score(y_train,ypred_train))
print('Cross validation score:',cross_val_score(log_model,x_train,y_train,cv = 5, scoring = 'accuracy'))
print('Test Accuracy:',accuracy_score(y_test,ypred_test))

print("")

from sklearn.neighbors import KNeighborsClassifier

estimator = KNeighborsClassifier()
param_grid = {'n_neighbors': list(range(1,50))}

from sklearn.model_selection import GridSearchCV
knn_grid = GridSearchCV(estimator, param_grid,scoring ='accuracy',cv = 5)
knn_grid.fit(x_train,y_train)

knn_model = knn_grid.best_estimator_

ypred_train = knn_model.predict(x_train)
ypred_test = knn_model.predict(x_test)

print('Train Accuracy:',accuracy_score(y_train,ypred_train))
print('Cross validation score:',cross_val_score(knn_model,x_train,y_train,cv = 5, scoring = 'accuracy'))
print('Test Accuracy:',accuracy_score(y_test,ypred_test))


print("")

from sklearn.tree import DecisionTreeClassifier
estimator = DecisionTreeClassifier(random_state = rs)
param_grid = {'criterion':['gini','entropy'],'max_depth':list(range(1,16))}

from sklearn.model_selection import GridSearchCV
dt_grid = GridSearchCV(estimator, param_grid,scoring = 'accuracy',cv = 5)
dt_grid.fit(x_train,y_train)

dt = dt_grid.best_estimator_
dt_fi = dt.feature_importances_

index = [i for i,x in enumerate(dt_fi) if x>0]

x_train_dt = x_train.iloc[:,index]
x_test_dt = x_test.iloc[:,index]

dt.fit(x_train_dt,y_train)

ypred_train = dt.predict(x_train_dt)
ypred_test = dt.predict(x_test_dt)
print('Train Accuracy:',accuracy_score(y_train,ypred_train))
print('Cross validation score:',cross_val_score(dt,x_train,y_train,cv = 5, scoring = 'accuracy'))
print('Test Accuracy:',accuracy_score(y_test,ypred_test))


print("")

from sklearn.ensemble  import RandomForestClassifier
estimator = RandomForestClassifier(random_state = rs)
param_grid = {'n_estimators':list(range(1,51))}

from sklearn.model_selection import GridSearchCV
rf_grid = GridSearchCV(estimator, param_grid,scoring = 'accuracy',cv = 5)
rf_grid.fit(x_train,y_train)

rf = rf_grid.best_estimator_
rf_fi = rf.feature_importances_

index = [i for i,x in enumerate(rf_fi) if x>0]

x_train_rf = x_train.iloc[:,index]
x_test_rf = x_test.iloc[:,index]

rf.fit(x_train_rf,y_train)

ypred_train = rf.predict(x_train_rf)
ypred_test = rf.predict(x_test_rf)

print('Train Accuracy:',accuracy_score(y_train,ypred_train))
print('Cross validation score:',cross_val_score(rf,x_train,y_train,cv = 5, scoring = 'accuracy'))
print('Test Accuracy:',accuracy_score(y_test,ypred_test))


print("")


from sklearn.ensemble  import AdaBoostClassifier
estimator = AdaBoostClassifier(random_state = rs)
param_grid = {'n_estimators':list(range(1,51))}

from sklearn.model_selection import GridSearchCV
ab_grid = GridSearchCV(estimator, param_grid,scoring = 'accuracy',cv = 5)
ab_grid.fit(x_train,y_train)

ab = ab_grid.best_estimator_
ab_fi = ab.feature_importances_

index = [i for i,x in enumerate(ab_fi) if x>0]

x_train_ab = x_train.iloc[:,index]
x_test_ab = x_test.iloc[:,index]

ab.fit(x_train_ab,y_train)

ypred_train = ab.predict(x_train_ab)
ypred_test = ab.predict(x_test_ab)

print('Train Accuracy:',accuracy_score(y_train,ypred_train))
print('Cross validation score:',cross_val_score(ab,x_train,y_train,cv = 5, scoring = 'accuracy'))
print('Test Accuracy:',accuracy_score(y_test,ypred_test))


print("")


from sklearn.ensemble  import GradientBoostingClassifier
estimator = GradientBoostingClassifier(random_state = rs)
param_grid = {'n_estimators':list(range(1,50)),'learning_rate':[0.1,0.2,0.3,0.4,0.5,0.7,0.9]}

from sklearn.model_selection import GridSearchCV
gb_grid = GridSearchCV(estimator, param_grid,scoring = 'accuracy',cv = 5)
gb_grid.fit(x_train,y_train)

gb = gb_grid.best_estimator_
gb_fi = gb.feature_importances_

index = [i for i,x in enumerate(gb_fi) if x>0]

X_train_gb = x_train.iloc[:,index]
X_test_gb = x_test.iloc[:,index]

gb.fit(X_train_gb,y_train)

ypred_train = gb.predict(X_train_gb)
ypred_test = gb.predict(X_test_gb)

print('Train Accuracy:',accuracy_score(y_train,ypred_train))
print('Cross validation score:',cross_val_score(gb,x_train,y_train,cv = 5, scoring = 'accuracy'))
print('Test Accuracy:',accuracy_score(y_test,ypred_test))


print("")
'''
import pickle
from xgboost import XGBClassifier
estimator = XGBClassifier(random_state = rs)
param_grid = {'n_estimators':[10,20,40,1000],'max_depth':[3,4,5],'gamma':[0,0.15,0.3,0.5,1]}

from sklearn.model_selection import GridSearchCV
xgb_grid = GridSearchCV(estimator, param_grid,scoring = 'accuracy',cv = 5)
xgb_grid.fit(x_train,y_train)

xgb = xgb_grid.best_estimator_
xgb_fi = xgb.feature_importances_

index = [i for i,x in enumerate(xgb_fi) if x>0]

x_train_xgb = x_train.iloc[:,index]
x_test_xgb = x_test.iloc[:,index]

xgb.fit(x_train_xgb,y_train)

ypred_train = xgb.predict(x_train_xgb)
ypred_test = xgb.predict(x_test_xgb)

print('Train Accuracy:',accuracy_score(y_train,ypred_train))
print('Cross validation score:',cross_val_score(xgb,x_train,y_train,cv = 5, scoring = 'accuracy'))
print('Test Accuracy:',accuracy_score(y_test,ypred_test))


with open(f'Loan_Aooroval_Prediction.pkl', 'wb') as file:
        pickle.dump(xgb, file)