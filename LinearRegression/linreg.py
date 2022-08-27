"""
Linear regression
Mariana Castro Payns - A01706038
"""

#  Import necessary ibraries
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Specify the names in an list
columns = ["Class", "Alcohol", "Malic acid", "Ash", "Alcalinity", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoids", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Poline"]
# Open the file and create the data frame
df_load = pd.read_csv('wine.data',names = columns)

# Drop some columns
df_wine = df_load.drop(["Malic acid", "Ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoids", "Proanthocyanins", "OD280/OD315 of diluted wines"], axis = 1)
df = df_wine[["Class", "Color intensity", "Hue", "Alcalinity", "Poline"]].to_numpy()
# Fill column with 0
df = np.c_[df, np.ones(178)]

# Define Alcohol as expected value to return
exp = df_wine["Alcohol"].to_numpy()

# Global error
__error__ = [];

# Hypotesis function, recieves parameters and prediction df
def hyp(params, df):
    sum = 0
    for i in range(len(params)):
        # Evaluates a simple y = mx + b
        sum = sum + params[i]*df[i]
    return sum

# Cost function, also used to display errors in plot
def cost(params, df, exp):
    global __error__
    mse = 0
    
    for i in range(len(df)):
        # Obtain hypotesis
        h = hyp(params, df[i])
        print( "Hypotesis:  %f  Expected Value: %f " % (h,  exp[i])) 
        orig_e = h-exp[i]
        # Get MSE
        mse = +orig_e**2
    mean = mse/len(df)

    # Get mean of errors to determine accuracy of model
    __error__.append(mean)
    return mean

# Optimizer - Gradient Descending
def GradDes(params, df, exp, alpha):
    # The function reciebes the parameters, data frame, expeced values and
    # the learining rate which in this case will be names alpha
    temp = list(params)

    for j in range(len(params)):
        acum = 0
        for i in range(len(params)):
            e = hyp(params, df[i]) - exp[i]
            sumatory = e * df[i][j]
            acum = acum + sumatory
        alpha = alpha*(1/len(df))*acum
        temp[j] = params[j] - alpha
    # Return list with new parameters
    return temp  

# Arbitrary params and alpha to minimize error
params = [-8.38, 4, 5]
# Learning rate
alfa = 0.03

# Initial epoch value
epochs = 0

# Run and evaluare function until stablished minima is reached
while True:  
    oldparams = list(params)
    print (params)
    params = GradDes(params, df,exp,alfa)	
    error = cost(params, df, exp)  #only used to show errors
    print (params)
    print ("Error", error)
    epochs = epochs + 1
    if(oldparams == params or epochs == 30):   # local minima is found
        #print ("samples:")
        #print(df)
        print ("final params:")
        print (params)
        break
 
# Generate a graph of the errors/loss so we can how is the model behaving
plt.plot(__error__)
plt.show()