import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ElasticNetStochastic as NetSto
import scipy
from scipy.special import fresnel
from mpl_toolkits import mplot3d
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from mpl_toolkits.mplot3d import Axes3D
pio.renderers.default='browser'
import sklearn
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_squared_error

covid = pd.read_excel("CaliData_NEW.xlsx")
ncol = len(covid.columns)
covid = covid.values
#Looking at the data, the first column that has desired info is 
#column 12, so the size we want is ncol-12

nexp = ncol-12
deaths = []
iterations = ncol-2

for i in range(12,iterations):
    deaths_on_day = abs(sum(covid[:,(i+1)])-sum(covid[:,i]))
    deaths.append(deaths_on_day)
    


pos_deaths = [n for n in deaths if n>0]
Y = np.log(pos_deaths)
day = range(len(pos_deaths))*np.ones(len(pos_deaths)) 
#normalize data
day = day/day.max()
df = pd.DataFrame()
df['Day'] = day
df['logDeaths'] = Y
df.to_csv("DayVSlogDeaths_Cali.csv")
X1 = np.exp(-day/10)*np.sin(np.pi*day)
X2 = np.exp(-day/10)*day
X3 = np.log(day+.2)
#X1 = day
#X2 = day**2
#X3 = day**3
#X4 = day**4
X=np.array([X1,X2,X3]).T


len1 = int(.75*len(day))
len2 = len(day)
X_train = X[0:len1:,]
X_test = X[len1:len2:,]
Y_train = Y[0:len1]
Y_test = Y[len1:len2]

lambdas = np.array([0,1,2,3])
model = NetSto.ElasticNet(.01,0,0,100000)
model.fit(X_train,Y_train)
Y_pred = model.predict(X)
plt.scatter(day*len(pos_deaths),Y,color="blue") 
plt.plot(day*len(pos_deaths),Y_pred,color="orange")
plt.title("Log Deaths Per Day vs Day in California")
plt.ylabel('Log Deaths Per Day')
plt.xlabel('Days Since March 2 2020')
plt.xlabel('March 2 to November 15 2020 ')
plt.show()
r2 = model.getr2(Y,Y_pred)
print(r2)
print(model.getr2a(r2))
print(model.getMSE(Y,Y_pred))


X = day
X = X.reshape(len(X),1)
X_train = X[0:len1]
X_test = X[len1:len2]

SVRModel = SVR()
SVRModel.fit(X_train,Y_train)
Y_pred = SVRModel.predict(X)

plt.scatter(day*len(pos_deaths),Y,color="blue") 
plt.plot(day*len(pos_deaths),Y_pred,color="orange")
plt.title("Log Deaths Per Day vs Day in California")
plt.ylabel('Log Deaths Per Day')
plt.xlabel('Days Since March 2 2020')
plt.show()

print("SVR STATS")
Yave = np.mean(Y)*np.ones(len(Y))
TSS = sum((Y-Yave)**2)
RSS = sum((Y-Y_pred)**2)
R2=1-RSS/TSS
print(R2)
MSE = (1/len(Y))*sum(Y-Y_pred)**2
print(MSE)





covid = pd.read_csv("time_series_covid_19_deaths_US.csv")
ncol = len(covid.columns)
covid = covid.values
#Looking at the data, the first column that has desired info is 
#column 12, so the size we want is ncol-12

nexp = ncol-12
deaths = []
iterations = ncol-2

for i in range(12,iterations):
    deaths_on_day = abs(sum(covid[:,(i+1)])-sum(covid[:,i]))
    deaths.append(deaths_on_day)

pos_deaths = [n for n in deaths if n>0]
Y = np.log(pos_deaths)
day = range(len(pos_deaths))*np.ones(len(pos_deaths)) 

#normalize data
day = day/day.max()
df = pd.DataFrame()
df['Day'] = day
df['logDeaths'] = Y
df.to_csv("DayVSlogDeaths_USA.csv")
X1 = np.exp(-day/10)*scipy.special.fresnel(1.1*day)[1]
X2 = np.exp(-day/2)*np.log(2*day+.1)
#X2 = scipy.special.fresnel(day)[0]
#X1 = day
#X2=day**2
#X3=day**3
#X4 = day**2
X=np.array([X1,X2]).T

#convert back to normal scale on the x axis
plt.scatter(day*len(pos_deaths),Y,color="blue") 
plt.title("Log Deaths Per Day vs Day in USA")
plt.ylabel('Log Deaths Per Day')
plt.xlabel('Days Since February 28 2020')
plt.xlabel('February 28 to November 15 2020 ')
len1 = int(.75*len(day))
len2 = len(day)
X_train = X[0:len1:,]
X_test = X[len1:len2:,]
Y_train = Y[0:len1]
Y_test = Y[len1:len2]

model = NetSto.ElasticNet(.01,0,0,100000)
model.fit(X_train,Y_train)
Y_pred = model.predict(X)

plt.plot(day*len(pos_deaths),Y_pred,color="orange")

plt.show()
r2 = model.getr2(Y,Y_pred)
print(r2)
print(model.getr2a(r2))
print(model.getMSE(Y,Y_pred))


X = day
X = X.reshape(len(X),1)
X_train = X[0:len1]
X_test = X[len1:len2]

SVRModel = SVR()
SVRModel.fit(X_train,Y_train)
Y_pred = SVRModel.predict(X)
plt.scatter(day*len(pos_deaths),Y,color="blue") 
plt.title("Log Deaths Per Day vs Day in USA")
plt.ylabel('Log Deaths Per Day')
plt.xlabel('Days Since February 28 2020')
plt.plot(day*len(pos_deaths),Y_pred,color="orange")

print("SVR STATS")
Yave = np.mean(Y)*np.ones(len(Y))
TSS = sum((Y-Yave)**2)
RSS = sum((Y-Y_pred)**2)
R2=1-RSS/TSS
print(R2)
MSE = (1/len(Y))*sum(Y-Y_pred)**2
print(MSE)


    