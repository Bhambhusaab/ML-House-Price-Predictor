# from collections import dict_keys
# from collections import dict_keys

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error as mse

diabetes = datasets.load_diabetes()
print(diabetes)
# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])

print(diabetes.DESCR)
# diabetes_x = diabetes.data[:, np.newaxis, 2]
diabetes_x = diabetes.data
# print(diabetes_x)
diabetes_x_train = diabetes_x[:-30]
diabetes_x_test = diabetes_x[-30:]
# print("X_test", diabetes_x_test)
# print("x_train ", diabetes_x_train)
# print(len(diabetes_x_train), len(diabetes_x_test), len(diabetes_x))
diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]
# print(diabetes_y_test, len(diabetes_y_test))
# print(diabetes_y_train, len(diabetes_y_train))

# model = linear_model.Lasso()
# model = linear_model.PassiveAggressiveClassifier()
model = linear_model.LinearRegression()
# model = linear_model.Ridge()

model.fit(diabetes_x_train, diabetes_y_train)

diabetes_y_predict = model.predict(diabetes_x_test)

print("Mean Squared Error Is:", mse(diabetes_y_test, diabetes_y_predict))

print("Weight:", model.coef_)

print("Intercept:", model.intercept_)

# plt.scatter(diabetes_x_test, diabetes_y_test)
# plt.plot(diabetes_x_test, diabetes_y_predict)
#
# plt.show()
