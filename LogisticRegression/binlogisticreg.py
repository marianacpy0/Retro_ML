"""
Binary logistic regression
Mariana Castro Payns - A01706038
"""
#  Import necessary ibraries
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Global variable to store errors and display them on graph
__error__ = [];

# Specify the names of df in an list
columns = ["Variance","Skewness","Kurtosis","Entropy","Class"]

# Open the file and create the data frame
df_bank = pd.read_csv('data_banknote_authentication.txt',names = columns)

# Add columns to numpy
df = df_bank[["Variance","Skewness","Kurtosis","Entropy"]].to_numpy()

# Fill column with 1
df = np.c_[df, np.ones(1372)]

# Define varianle for expected values 
exp = df_bank["Class"].to_numpy()

# Hypotesis function to evaluate the linear function with parameters
def hyp(params, df):
    # Reciebes: parameters - for each element x of the df and
    # df contains the values of sample or data frame
    sum = 0
    for i in range(len(params)):
        # Evaluation of the h(x)
        sum = sum + (params[i]*df[i])
    sum = sum*(-1)
    # Sigmoid optimizer 
    sig = 1/(1+math.exp(sum))
    # it returns a value which indicates the hypotesis of the model
    return sig

# Cost function, primarly used to display error 
def cost(params, df, exp):
    # Recieves the parameters, the dataset and the expected values
    global __error__
    total_error = 0
    error = 0

    for i in range(len(df)):
        h = hyp(params, df[i])

        # Avoid the log(0) error
        if(exp[i] == 1):
            if(h == 0):
                h = 0.0001
            error = (-1)*math.log(h)
        if(exp[i] == 0):
            if(h == 1):
                h = 0.9999
            error = (-1)*math.log(1 - h)
        print( "Error: %f  Hypotesis:  %f  Expected Value: %f " % (error, h,  exp[i]))
        total_error = total_error + error
    mean = total_error/len(df)
    # Obtain mean of the errors to get the error of the model
    __error__.append(mean)
    return mean

# Optimization function using Gradient Descending
def GradDesc(params, df, exp, alpha):
    # The function reciebes the parameters, data frame, expeced values and
    # the learining rate which in this case will be names alpha
    oldparams = list(params)
    # Apply the Gradient Descending theorical formula 
    for j in range(len(params)):
        sum = 0
        for i in range(len(df)):
            hypot = hyp(params, df[i])
            error = hypot - exp[i]
            lr = error*df[i][j]
            sum = sum + lr # Sum part of de GD formula
        subs = alpha*(1/len(df))*sum
        oldparams[j] = params[j] - subs
    # Return the old parameters
    return oldparams


# Definition of params and learning rate
params = [-0.8,1.6,0.5]
alpha = 0.1  

# Run and call GD and costs function until minima is reached
while True: 
    oldparams = list(params)
    print (params)
    params=GradDesc(params, df,exp,alpha)
    error = cost(params, df, exp) #To show errors
    print("Error: ", error)
    print ("Params: ", params)
    if(oldparams == params or error < 0.2): # in case local minima is found 
        print ("Final Params: ")
        print (params)
        break

# Plot result for errors
plt.plot(__error__)
plt.show()
