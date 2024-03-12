import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

from sklearn import linear_model
from sklearn.metrics import r2_score


# df = pd.read_csv("./fuel_consumption.csv")
df = pd.read_csv("./FuelConsumption.csv")

# take a look at the dataset
df.head()

# summarize the data
df.describe()

# select specific columns - features
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

# show total histogram for this features
viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()

# show linear for current columns
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
  # give names for x and y
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

# show linear for current columns
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='red')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


### TASK
# Plot CYLINDER vs the Emission, to see how linear is their relationship is:
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color="green")
plt.xlabel("Cylinders")
plt.ylabel("Emission")
plt.show()


# Creating train and test data sets
# mask data train:80% test:20%
msk = np.random.rand(len(df)) < 0.8

# !!! to use this syntax you need to convert list to np array:: list = np.array(list) 
train = cdf[msk]
test = cdf[~msk]

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='red')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# TRAINING DATA
# Modeling with sklearn package 
regr = linear_model.LinearRegression()

train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

regr.fit(train_x, train_y)

# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

# plot the results
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")


# TESTING DATA
# Evaluation 
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
# predicted value
test_y_ = regr.predict(test_x)

# output Result
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )



# EXERCISE
# Lets see what the evaluation metrics are if we trained a regression model using the FUELCONSUMPTION_COMB feature

# set training and test data for x for needed feature
train_x = train[['FUELCONSUMPTION_COMB']]
test_x = test[['FUELCONSUMPTION_COMB']]

# init Simple Linear Regression
regr = linear_model.LinearRegression()

# model with trained data
regr.fit(train_x, train_y)

predict = regr.predict(test_x)

# output absolute error
print("Mean absolute error: %.2f" % np.mean(np.absolute(predict - test_y)))