import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


ds=pd.read_csv("salary.csv")

ds.head()

ds.dtypes

x=ds.iloc[:,:-1].values
y=ds.iloc[:,-1].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import Ridge,Lasso

rd=Ridge(alpha=3)
rd.fit(x_train,y_train)
rd.score(x_test,y_test)

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,rd.predict(x_train),color='blue')
plt.title("Salary vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")

ls=Lasso(alpha=3)
ls.fit(x_train,y_train)
ls.score(x_test,y_test)

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,ls.predict(x_train),color='blue')
plt.title("Salary vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")