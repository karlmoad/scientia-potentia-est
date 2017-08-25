# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----Begin Data Preprocessing
# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

# setup our independat features for evaluation 
# use columns creditscore thru Estimated salary 
X = dataset.iloc[:, 3:13].values

#set our dependant feature stating if customer left bank "exited"
y = dataset.iloc[:, 13].values

# Encoding categorical data
labelencoder_X_country = LabelEncoder()
X[:, 1] = labelencoder_X_country.fit_transform(X[:, 1])

labelencoder_X_gender = LabelEncoder()
X[:, 2] = labelencoder_X_gender.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# drop one of the categories to eliminate dummy variable trap 
# esseutually each category trns into a colum indicating state 0,1 problme occurs
# when the intercept of each row of the matrix = 1 meaning that they each infer the other hence
# will skew the measurement in the linear calc, so we drop one of the category columns to keep that form happening
X = X[: , 1:]

# Splitting the dataset into the Training set and Test set
# X = 10000 rows so .2 test size will = 2000 for test leaving 8000 for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# ----End Data Preprocessing

# ----Begin ANN Build


# ----End ANN 

# ----Begin Make predictions and evaluating model

# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)