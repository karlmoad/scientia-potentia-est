# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
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
# init the ANN
ANN = Sequential() 
# add input layer and first hidden layer
# create a dense layer CMD + i to see documentation
# output_dim = number of nodes in firts hidden layer use tip
#        number of input layer nodes + output layer nodes / 2 to get average
#        so ther are 11 in and 1 out 11 + 1 = 12 /2 = 6  
ANN.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))

# add a second hidden layer 
ANN.add(Dense(output_dim=6,init='uniform',activation='relu'))

# add output layer
# binary output will only need a single node
ANN.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

# compile the ANN
# adam = Stocastic Gradient Decent Model algorithm
ANN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# fitting ANN to the training set
# batch size = number of steps to bacth into an update 
ANN.fit(X_train, y_train, batch_size=10, epochs=100)

# ----End ANN 

# ----Begin Make predictions and evaluating model

# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
y_pred = ANN.predict(X_test)

# derive true or false based on probability predicted
y_pred = (y_pred > 0.5)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

""" 
Predict if a customer will leave the bank with the following info
-----------------------------------------------------------------
Geography: France
Credit Scroe: 600
Gender: Male
Age:40
Tenure: 3
Balance: 60000
Number of Products: 2
Credit Card: Yes
Active: Yes
Salary: 50000
"""

new_record = np.array([[0.0,0.0,600.0,1.0,40.0,3.0,60000.0,2.0,1.0,1.0,50000.0]])
new_rec = sc.transform(new_record)

new_prediction = ANN.predict(new_rec)

print("New Customer will exit: {}".format(new_prediction > 0.5))

# Evaluate, Imporve, Tune using KFold cross validation

def build_classifier():
    # Buld ANN Model like above
    classifier = Sequential() 
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator=classifier, X = X_train, y = y_train, cv=10, n_jobs=-1)













ANN.save("model", overwrite=True, include_optimizer=True)






