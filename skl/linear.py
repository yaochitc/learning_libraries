import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

sample_size = 1000
feature_size = 3

weights = np.array([[1.0], [2.0], [3.0]])
bias = 2.0
x = np.random.rand(sample_size, feature_size)
y = np.matmul(x, weights) + bias

train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8)

regressor = LinearRegression()
regressor.fit(train_x, train_y)

print(mean_squared_error(test_y, regressor.predict(test_x)))