import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

from sklearn import linear_model
from sklearn.metrics import r2_score


# Example of Simpler Linear Regression

# get data from CSV
df = pd.read_csv("./FuelConsumption.csv")

# get required features (columns)
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
 
# create training and test data
msk = np.random.rand(len(df)) < 0.8 # 80 to 20

train = cdf[msk] # will get 80% who are true
test = cdf[~msk] # will get 20% who are false and converted to true

# create regression model
regr = linear_model.LinearRegression()

# prepare training data 
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

# run regression model
regr.fit(train_x, train_y)

# get Coefficient and Intercept
coef = regr.coef_
inter =regr.intercept_

# plot the fit line
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='green')
plt.plot(train_x, coef[0][0]*train_x + inter[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# test model with test data
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

# create prediction for y using x
test_y_ = regr.predict(test_x)

# Output results
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )