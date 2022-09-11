"""
Logistic regression with frameworks
Mariana Castro Payns - A01706038
"""

#  Import necessary ibraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# First we choose a class of model and model hyperparameters.
# for this logistic regression we specify intercept
model = LogisticRegression(fit_intercept=True)
print(model)

#load data set
df_wine = pd.read_csv('winequality-red.csv')

# Define quality as expected value to return
exp = df_wine['quality']
X = df_wine.drop(['quality'], axis = 1)
# Check data and expected size
print("Size of DF - ", X.shape)
print("Size of expected values - ", exp.shape)

# Split data
Xtrain, Xtest, ytrain, ytest = train_test_split(X, exp)

# Specify model with maximun iterations
model = LogisticRegression(max_iter=200000)

# Fit model to data
model.fit(Xtrain, ytrain)

# Menu to get multiple predictions
ans=True
while ans:
    print("""
    1. Show model data (parameters, bias and score)
    2. Prediction 1
    3. Prediction 2
    4. Prediction 3
    5. Prediction 4
    6. Prediction 5
    7. Exit
    """)
    ans=input("What would you like to do? ")
    if ans=="1":
        # Parameters 
        print("Pameters of model: ", model.coef_)
        # Bias value
        print("Bias of model: ", model.intercept_)
        # Score of model
        print("Score: ", model.score(Xtest, ytest))
    elif ans=="2":
        # Prediction 1, for all data frame
        predicted1 = model.predict(Xtest)
        print("Expected Values: ", exp)
        print("Predictions for wine quality data set: ", predicted1)
    elif ans=="3":
        # Prediction 2 for the first row of df
        xfit = X.iloc[:1, :]
        predicted2 = model.predict(xfit)
        print("Expected value: ", exp.iloc[1])
        print("Predictions: ", predicted2)
    elif ans=="4":
        # Prediction 3 for the 25th and 35th rows of df
        xfit1 = X.iloc[[25,35],:]
        predicted3 = model.predict(xfit1)
        print("Expected value: ", exp.iloc[25])
        print("Expected value: ", exp.iloc[35])
        print("Predictions: ", predicted3)
    elif ans=="5":
        # Prediction 4 for the 52, 22, 12 rows of df
        xfit2 = X.iloc[[52,22,12],:]
        predicted4 = model.predict(xfit2)
        print("Expected value: ", exp.iloc[52])
        print("Expected value: ", exp.iloc[22])
        print("Expected value: ", exp.iloc[12])
        print("Predictions: ", predicted4)
    elif ans=="6":
        # Prediction 5 for the 158, 147, 236, 12 rows of df
        xfit3 = X.iloc[[158,147,236,12],:]
        predicted5 = model.predict(xfit3)
        print("Expected value: ", exp.iloc[158])
        print("Expected value: ", exp.iloc[147])
        print("Expected value: ", exp.iloc[236])
        print("Expected value: ", exp.iloc[12])
        print("Predictions: ", predicted5)
    elif ans=="7":
        # Exit menu
        print("\n Goodbyeeeeeeeeeeee") 
        ans = None
    else:
        # Not valid option.
        print("\n Not Valid Choice Try again")

