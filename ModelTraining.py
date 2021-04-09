#import necessary libraries
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import pickle


data = pd.read_csv("dataset/diabetes.csv")

#performing EDA

data["Glucose"] = data["Glucose"].replace(0,data["Glucose"].mean())
data["BloodPressure"] = data["BloodPressure"].replace(0,data["BloodPressure"].mean())
data["BMI"] = data["BMI"].replace(0,data["BMI"].mean())
data["SkinThickness"] = data["SkinThickness"].replace(0,data["SkinThickness"].mean())
data["Insulin"] = data["Insulin"].replace(0,data["Insulin"].mean())

#removing top 10% data of insulin
out = data["Insulin"].quantile(0.90)
data = data[data["Insulin"]<out]

#removing top !% data of Pregnancies
out = data["Pregnancies"].quantile(0.99)
data = data[data["Pregnancies"]<out]

#removing top 2% data of BloodPressure column
out = data["BloodPressure"].quantile(0.98)
data = data[data["BloodPressure"]<out]

#removing top 2% data of Skinthickness column
out = data["SkinThickness"].quantile(0.98)
data = data[data["SkinThickness"]<out]

#removing top 2% data from BMI column
out = data["BMI"].quantile(0.98)
data = data[data["BMI"]<out]

#removing top 1% from Age column
out = data["Age"].quantile(0.99)
data = data[data["Age"]<out]

features = data.drop("Outcome",axis=1)
y = data["Outcome"]

scaling = StandardScaler()
scaled_features = scaling.fit_transform(features)

train_x,test_x,train_y,test_y = train_test_split(scaled_features,y,test_size=0.25,random_state=355)

knn = KNeighborsClassifier()
knn.fit(train_x,train_y)

print("Goodness of the model:", knn.score(test_x,test_y))

#lets evaluate testing score of our model
print(accuracy_score(test_y,knn.predict(test_x)))



filename = "diabetes-prediction.pkl"
pickle.dump(knn,open(filename,'wb'))
print("<<<<< Successfully saved model file >>>>>>")