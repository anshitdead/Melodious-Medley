#importing the libraries
import pandas as pd
import numpy as np
import lightgbm as lgb
#importing the datasets
dataset=pd.read_csv('train.csv')
dataset2=pd.read_csv('test.csv')
#Using Time-Series Data
dataset['ts_listen']=pd.to_datetime(dataset.ts_listen)
dataset['Weekday']=dataset.ts_listen.dt.weekday

dataset2['ts_listen']=pd.to_datetime(dataset2.ts_listen)
dataset2['Weekday']=dataset2.ts_listen.dt.weekday
#Making Dummy Variables for Weekday
dataset = pd.get_dummies(dataset, columns=['Weekday'], drop_first=True)
dataset2 = pd.get_dummies(dataset2, columns=['Weekday'], drop_first=True)
#Extracting features from dataset to variables X, y, X_test
X_test = dataset2.iloc[:, [0,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20]].values
X_test = pd.DataFrame(X_test)
X = dataset.iloc[:, [0,2,3,4,5,6,7,8,9,10,11,12,13,16,17,18,19,20,21]].values
X = pd.DataFrame(X)
y = dataset.iloc[:, 14].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_valid = sc.transform(X_valid)

# Importing the Keras libraries and packages
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 30, activation = 'relu', input_dim = 19))
classifier.add(Dropout(0.2))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 30, activation = 'relu'))
classifier.add(Dropout(0.2))

# Adding the third hidden layer
classifier.add(Dense(output_dim = 30, activation = 'relu'))
classifier.add(Dropout(0.2))

# Adding the fourth hidden layer
classifier.add(Dense(output_dim = 30, activation = 'relu'))
classifier.add(Dropout(0.2))

# Adding the fifth hidden layer
classifier.add(Dense(output_dim = 30, activation = 'relu'))
classifier.add(Dropout(0.2))

# Adding the output layer
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=16, epochs=10, validation_data=(X_valid, y_valid), shuffle=True)


Making the predictions and evaluating the model
score = classifier.evaluate(X_valid, y_valid)
print(score[1])
# Predicting the Test set results
y_pred1 = classifier.predict(X_test)

#LIGHTGBM MODEL
train_data=lgb.Dataset(X_train,label=y_train)
#setting parameters for lightgbm
param = {'boosting_type': 'gbdt',
#          'max_depth' : -1,
          'objective': 'binary',
#          'nthread': 3, # Updated from nthread
#          'num_leaves': 20,
#          'learning_rate': 0.1,
#          'max_bin': 1000,
#          'subsample_for_bin': 200,
#          'subsample': 1,
#          'subsample_freq': 1,
#          'colsample_bytree': 0.8,
#          'reg_alpha': 1,
#          'reg_lambda': 0.01,
#          'min_split_gain': 0.5,
#          'min_child_weight': 1,
#          'min_child_samples': 5,
#          'scale_pos_weight': 1,
#          'num_class' : 1,
          'metric' : ['auc', 'binary_logloss']}
#training our model using light gbm
num_round=100
lgbm=lgb.train(param,train_data,num_round)
#predicting on test set
ypred=lgbm.predict(X_valid)
#converting probabilities into 0 or 1
for i in range(0,126457):
    if ypred[i]>=.5:       # setting threshold to .5
       ypred[i]=1
    else:  
       ypred[i]=0

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
#calculating accuracy
accuracy_lgbm = accuracy_score(ypred,y_valid)
accuracy_lgbm
#calculating roc_auc_score for light gbm. 
auc_lgbm = roc_auc_score(y_valid,ypred)
auc_lgbm
#Setting the Grid Parameters
gridParams = {
    'learning_rate': [0.5],
    'n_estimators': [200],
    'num_leaves': [100,150,200,250,300],
    'boosting_type' : ['gbdt'],
    'objective' : ['binary'],
    'max_bin' : [1100]
#    'random_state' : [501], # Updated from 'seed'
#    'colsample_bytree' : [0.65, 0.66],
#    'subsample' : [0.7,0.75],
#    'reg_alpha' : [1,3,5,7,9],
#    'reg_lambda' : [0.01,0.03,0.05,0.07]
    }

mdl = lgb.LGBMClassifier(boosting_type= 'gbdt',
          objective = 'binary',
#          n_jobs = 3, # Updated from 'nthread'
#          silent = True,
#          max_depth = param['max_depth'],
          max_bin = param['max_bin'],
          learning_rate = param['learning_rate'],
          num_leaves = param['num_leaves']
#          subsample_for_bin = param['subsample_for_bin'],
#          subsample = param['subsample'],
#          subsample_freq = param['subsample_freq'],
#          min_split_gain = param['min_split_gain'],
#          min_child_weight = param['min_child_weight'],
#          min_child_samples = param['min_child_samples'],
#          scale_pos_weight = param['scale_pos_weight']
)

# Create the grid
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(mdl, gridParams,
                    verbose=0,
                    cv=10,
                    n_jobs=-1)
# Run the grid
grid.fit(X_train, y_train)


print('Best parameters found by grid search are:', grid.best_params_)
print('Best score found by grid search is:', grid.best_score_)

y_pred2=lgbm.predict_proba(X_test)
y_pred=(y_pred1+y_pred2)/2
#converting probabilities into 0 or 1
y_pred = (y_pred > 0.5).astype(int)
       
       
