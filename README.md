# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file

# PROGRAM
```
Developed By : Vijay Ganesh N
Reg No : 212221040177
```
- <B>DATA PREPROCESSING BEFORE FEATURE SELECTION:</B>
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/titanic_dataset.csv')
df.head()
```
![280198608-30c4abf7-02ab-456d-9fbe-4aaa6303d1b4](https://github.com/Aakash0407/ODD2023-Datascience-Ex-07/assets/118799103/699c8540-af95-41a8-8363-600501fbc89d)
- <B>CHECKING NULL VALUES:</B>
```python
df.isnull().sum()
```
![280198818-958ca867-e8bc-4f0c-9c30-6e9b8f0ae92d](https://github.com/Aakash0407/ODD2023-Datascience-Ex-07/assets/118799103/d9a1babc-6583-4b26-8e12-385152df5681)
- <B>DROPPING UNWANTED DATAS:</B>
```python
df.drop('Cabin',axis=1,inplace=True)
df.drop('Name',axis=1,inplace=True)
df.drop('Ticket',axis=1,inplace=True)
df.drop('PassengerId',axis=1,inplace=True)
df.drop('Parch',axis=1,inplace=True)
df.head()
```
![280199043-772589ff-b08f-47f4-bf51-5a1a4776b39d](https://github.com/Aakash0407/ODD2023-Datascience-Ex-07/assets/118799103/8696c012-268a-41ea-ad6a-c6c9f9284702)
- <B>DATA CLEANING:</B>
```python
df['Age']=df['Age'].fillna(df['Age'].median())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df.isnull().sum()
```
![280201596-d9368867-c4ec-4da0-b947-e9ca1d57a076](https://github.com/Aakash0407/ODD2023-Datascience-Ex-07/assets/118799103/565cabc6-6e84-48b6-9e68-4fd2589c85f8)
- <B>REMOVING OUTLIERS:</B>
  - Before
```python
plt.title("Dataset with outliers")
df.boxplot()
plt.show()
```
![280202248-2800a653-f79a-477a-81ed-e1c0096e0fb4](https://github.com/Aakash0407/ODD2023-Datascience-Ex-07/assets/118799103/e1700201-544a-498e-ae86-37641a9486ad)
- 
  - After
```python
cols = ['Age','SibSp','Fare']
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
plt.title("Dataset after removing outliers")
df.boxplot()
plt.show()
```
![280202684-f7fc0e45-822a-4529-94e1-a9f153e97071](https://github.com/Aakash0407/ODD2023-Datascience-Ex-07/assets/118799103/ea76c888-d460-49b2-93c5-a93055ed4447)
- <B>_FEATURE SELECTION:_</B>
```python
from sklearn.preprocessing import OrdinalEncoder
climate = ['C','S','Q']
en= OrdinalEncoder(categories = [climate])
df['Embarked']=en.fit_transform(df[["Embarked"]])
df.head()
```
![280203177-5ac83c7e-75c7-436d-9fe8-35049a1dfe8e](https://github.com/Aakash0407/ODD2023-Datascience-Ex-07/assets/118799103/04948384-773c-4bbb-ae78-a36dae060e63)
```python
from sklearn.preprocessing import OrdinalEncoder
gender = ['male','female']
en= OrdinalEncoder(categories = [gender])
df['Sex']=en.fit_transform(df[["Sex"]])
df.head()
```
![280203425-48a333b2-188a-4721-9474-936e846b9d77](https://github.com/Aakash0407/ODD2023-Datascience-Ex-07/assets/118799103/9d6502cb-b339-4b8c-b605-ee37dc8e5433)
```python
from sklearn.preprocessing import RobustScaler
sc=RobustScaler()
df=pd.DataFrame(sc.fit_transform(df),columns=['Survived','Pclass','Sex','Age','SibSp','Fare','Embarked'])
df.head()
```
![280203705-328828f8-440b-4466-a227-b6d47614e9da](https://github.com/Aakash0407/ODD2023-Datascience-Ex-07/assets/118799103/9521fefa-4bb5-4099-82ed-44f2200de445)
```python
import statsmodels.api as sm
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal',n_quantiles=692)

df1=pd.DataFrame()
df1["Survived"]=np.sqrt(df["Survived"])
df1["Pclass"],parameters=stats.yeojohnson(df["Pclass"])
df1["Sex"]=np.sqrt(df["Sex"])
df1["Age"]=df["Age"]
df1["SibSp"],parameters=stats.yeojohnson(df["SibSp"])
df1["Fare"],parameters=stats.yeojohnson(df["Fare"])
df1["Embarked"]=df["Embarked"]
df1.skew()
```
![280203793-45a00872-fa32-47da-909f-21d4bf633edf](https://github.com/Aakash0407/ODD2023-Datascience-Ex-07/assets/118799103/b4961be8-6dc1-4634-a74c-39fb78b4de35)
```python
import matplotlib
import seaborn as sns
import statsmodels.api as sm
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

X = df1.drop("Survived",1) 
y = df1["Survived"] 
```

- <B>_FILTER METHOD:_</B>
```python
plt.figure(figsize=(7,6))
cor = df1.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.RdPu)
plt.show()
```
![280204212-85f54298-d3e4-4c2a-b7a4-dd864908379a](https://github.com/Aakash0407/ODD2023-Datascience-Ex-07/assets/118799103/888aa8fa-c655-439b-98ee-a6cec924c868)
- <B>_HIGHLY CORRELATED FEATURES WITH THE OUTPUT VARIABLE SURVIVED:_</B>
```python
cor_target = abs(cor["Survived"])
relevant_features = cor_target[cor_target>0.5]
relevant_features
```
![280204862-f725dd10-f2e8-4e11-9f2e-b6ab73ac6135](https://github.com/Aakash0407/ODD2023-Datascience-Ex-07/assets/118799103/c70479d9-14b7-4e97-b88c-29cdfa4884b9)
- <B>_BACKWARD ELIMINATION:_</B>
```python
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
```
![280204975-951dc4a9-a94e-49ae-a7cc-0d28fb68e937](https://github.com/Aakash0407/ODD2023-Datascience-Ex-07/assets/118799103/97d566f3-c91c-403e-a5c5-46848b90ef4a)
- <B>_OPTIMUM NUMBER OF FEATURES THAT HAVE HIGH ACCURACY:_</B>
```python
nof_list=np.arange(1,6)            
high_score=0
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,step=nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))
```
![280207316-5eece329-e9b4-44f0-9411-29d992453afc](https://github.com/Aakash0407/ODD2023-Datascience-Ex-07/assets/118799103/85061da7-2b09-456b-b7e5-6d169253991c)
- <B>_FINAL SET OF FEATURE:_</B>
```python
cols = list(X.columns)
model = LinearRegression()
rfe = RFE(model, step=2)             
X_rfe = rfe.fit_transform(X,y)  
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)
```
![280207489-7ec9be2c-f22d-4adc-a959-be58866826d7](https://github.com/Aakash0407/ODD2023-Datascience-Ex-07/assets/118799103/d20fdb6c-a8fc-4a11-ab75-9b7e318dd4d0)
- <B>_EMBEDDED METHOD:_</B>
```python
reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (5.0, 5.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
plt.show()
```
![280207669-728772fb-2d29-4eb2-bcff-a383dbd125f4](https://github.com/Aakash0407/ODD2023-Datascience-Ex-07/assets/118799103/079c34f5-e554-4b38-92fa-0481cfb4a942)

# RESULT:
Thus, the various feature selection techniques have been performed on a given dataset successfully.
