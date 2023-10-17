# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn
## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

### Step 1
Prepare your data
Clean and format your data
Split your data into training and testing sets
### Step 2

Define your model
Use a sigmoid function to map inputs to outputs
Initialize weights and bias terms
### Step 3
Define your cost function
Use binary cross-entropy loss function
Penalize the model for incorrect predictions

### Step 4
Define your learning rate
Determines how quickly weights are updated during gradient descent

### Step 5
Train your model
Adjust weights and bias terms using gradient descent
Iterate until convergence or for a fixed number of iterations

### Step 6
Evaluate your model
Test performance on testing data
Use metrics such as accuracy, precision, recall, and F1 score

### Step 7
Tune hyperparameters
Experiment with different learning rates and regularization techniques

### Step 8
Deploy your model
Use trained model to make predictions on new data in a real-world application.

## Program:
```py
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: RAKESH JS
RegisterNumber: 212222230115
*/

import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
### Initial data set:
![image](https://github.com/MukeshVelmurugan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707363/fece68b1-d00d-4ea2-8e08-cb6f15bac397)

### Data info:
![image](https://github.com/MukeshVelmurugan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707363/b515f41e-f9bd-419d-ab87-9e9a0067a618)

### Optimization of null values:
![image](https://github.com/MukeshVelmurugan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707363/3238ef75-0ed5-4fef-b346-bf637b52d659)

### Assignment of x and y values:
![image](https://github.com/MukeshVelmurugan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707363/064ce9d9-aadc-4417-9271-642ac679f330)
![image](https://github.com/MukeshVelmurugan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707363/71208bc0-66ce-4d89-924c-1c55ef1a62d2)

### Converting string literals to numerical values using label encoder:
![image](https://github.com/MukeshVelmurugan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707363/3bc2e1c8-78e4-476b-8f46-5b91915f98b5)

### Accuracy:

![image](https://github.com/MukeshVelmurugan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707363/554b33d9-0e67-470b-b32d-b18d61735f12)

### Prediction:
![image](https://github.com/MukeshVelmurugan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707363/45ce7119-f0e2-45e9-b067-2c1ebece50ef)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
