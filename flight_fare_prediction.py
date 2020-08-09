# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 17:02:02 2020

@author: rkrit
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

train_data = pd.read_excel('Data_Train_flight.xlsx')

pd.set_option('display.max_columns',None) #displays all columns

train_data.dropna(inplace=True)
#EDA
#Date of Journey

train_data["Journey_Date"] = pd.to_datetime(train_data.Date_of_Journey , format = "%d/%m/%Y").dt.day
train_data["Journey_Month"] = pd.to_datetime(train_data.Date_of_Journey , format = "%d/%m/%Y").dt.month

train_data.drop("Date_of_Journey", axis=1, inplace=True)

#Departure time

train_data["Dep_hour"] = pd.to_datetime(train_data.Dep_Time).dt.hour
train_data["Dep_minutes"] = pd.to_datetime(train_data.Dep_Time).dt.minute

train_data.drop("Dep_Time", axis=1, inplace=True)

#arrival time

train_data["Arrival_hour"] = pd.to_datetime(train_data.Arrival_Time).dt.hour
train_data["Arrival_minutes"] = pd.to_datetime(train_data.Arrival_Time).dt.minute

train_data.drop("Arrival_Time", axis=1, inplace=True)

# Assigning and converting Duration column into list
duration = list(train_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration
    
# Adding duration_hours and duration_mins list to train_data dataframe

train_data["Duration_hours"] = duration_hours
train_data["Duration_mins"] = duration_mins

train_data.drop(["Duration"], axis = 1, inplace = True)

#Handling Categorial Data

#Airline
airline = train_data[["Airline"]]
airline = pd.get_dummies(airline, drop_first = True)

#Source
source = train_data[["Source"]]
source = pd.get_dummies(source , drop_first= True)

#Destination
destination = train_data[["Destination"]]
destination = pd.get_dummies(destination, drop_first= True)


#dropping routes and additional_info because no necessary info

train_data.drop(["Route","Additional_Info"],axis=1,inplace= True)

#Total Stops

train_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)

# Concatenate dataframe --> train_data + Airline + Source + Destination

data_train = pd.concat([train_data, airline, source, destination], axis = 1)

data_train.drop(["Airline","Source","Destination"],axis=1,inplace=True)

#Test data

Test_data = pd.read_excel("Test_set_flight.xlsx")
 
#Date of Journey

Test_data["Journey_Date"] = pd.to_datetime(Test_data.Date_of_Journey , format = "%d/%m/%Y").dt.day
Test_data["Journey_Month"] = pd.to_datetime(Test_data.Date_of_Journey , format = "%d/%m/%Y").dt.month

Test_data.drop("Date_of_Journey", axis=1, inplace=True)

#Departure time

Test_data["Dep_hour"] = pd.to_datetime(Test_data.Dep_Time).dt.hour
Test_data["Dep_minutes"] = pd.to_datetime(Test_data.Dep_Time).dt.minute

Test_data.drop("Dep_Time", axis=1, inplace=True)

#arrival time

Test_data["Arrival_hour"] = pd.to_datetime(Test_data.Arrival_Time).dt.hour
Test_data["Arrival_minutes"] = pd.to_datetime(Test_data.Arrival_Time).dt.minute

Test_data.drop("Arrival_Time", axis=1, inplace=True)

# Assigning and converting Duration column into list
duration = list(Test_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration

# Adding duration_hours and duration_mins list to train_data dataframe

Test_data["Duration_hours"] = duration_hours
Test_data["Duration_mins"] = duration_mins

Test_data.drop(["Duration"], axis = 1, inplace = True)

#Handling Categorial Data

#Airline
airline = Test_data[["Airline"]]
airline = pd.get_dummies(airline, drop_first = True)

#Source
source = Test_data[["Source"]]
source = pd.get_dummies(source , drop_first= True)

#Destination
destination = Test_data[["Destination"]]
destination = pd.get_dummies(destination, drop_first= True)


#dropping routes and additional_info because no necessary info

Test_data.drop(["Route","Additional_Info"],axis=1,inplace= True)

#Total Stops

Test_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)

# Concatenate dataframe --> train_data + Airline + Source + Destination

data_test = pd.concat([Test_data, airline, source, destination], axis = 1)

data_test.drop(["Airline","Source","Destination"],axis=1,inplace=True)


#Feature Selection
#defining independent and dependent features

y= data_train.iloc[:,1]
data_train.drop('Price',axis=1,inplace=True)
X=data_train

# Finds correlation between Independent and dependent attributes

plt.figure(figsize = (18,18))
sns.heatmap(train_data.corr(), annot = True, cmap = "RdYlGn")

plt.show()

# Important feature using ExtraTreesRegressor

from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(X, y)


plt.figure(figsize = (12,8))
feat_importances = pd.Series(selection.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()

#K best
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns

featureScores

print(featureScores.nlargest(10,'Score'))  #print 10 best features

#Machine Learning Model - Random Forest

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


from sklearn.ensemble import RandomForestRegressor
reg_rf = RandomForestRegressor()
reg_rf.fit(X_train, y_train)

y_pred = reg_rf.predict(X_test)

reg_rf.score(X_train, y_train)

reg_rf.score(X_test, y_test)

sns.distplot(y_test-y_pred)
plt.show()

plt.scatter(y_test, y_pred, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

metrics.r2_score(y_test, y_pred)

rf=RandomForestRegressor(n_estimators=700,min_samples_split= 15,min_samples_leaf= 1,max_features= 'auto',max_depth= 20)

rf.fit(X_train,y_train)

prediction = rf.predict(X_test)

plt.figure(figsize = (8,8))
plt.scatter(y_test, prediction, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()

import pickle
# open a file, where you ant to store the data
file = open('flight_fare_prediction_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf, file)

model = open('flight_fare_prediction_model.pkl','rb')
forest = pickle.load(model)


y_prediction = forest.predict(X_test)

metrics.r2_score(y_test, y_prediction)
