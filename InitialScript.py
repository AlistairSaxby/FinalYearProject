import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("PYTHONDATA.csv", sep=",")

print(data.head())  # A test to ensure the data has been read in correctly by printing first 5 rows

predict = "HBVDNACopies"
feature = "HbsAg"

X = np.array(data.drop([predict], 1))  # separating the Feature what we are using to predict by dropping the Label
y = np.array(data[predict])  # separating the Label what we are trying to predict
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)  # splits data 10% testing 90% training

bestaccuracy = 0
for _ in range(
        30):  # for loop will repeat regression 100 times only saving if new model is more accurate than previous model this is good practice

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)

    if acc > bestaccuracy:
        bestaccuracy = acc
        # print('Accuracy: ', acc)
        with open("LinearRegressionModel.pickle", "wb") as f:
            pickle.dump(linear, f)  # saves a pickle file of the model

print('Coefficient: ', linear.coef_)  # These are each slope value
print('Intercept: ', linear.intercept_)  # This is the intercept
    
# Testing the model
predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print("Calculated HBV Value =", predictions[x], ". HBsAg Value =", x_test[x], ". Actual HBV Value =",
          y_test[x])  # Where the variables are in order - Actual HBV Value


style.use("ggplot")
pyplot.scatter(data[predict], data[feature])
pyplot.xlabel("HBV DNA Copies Value")
pyplot.ylabel("HBsAg Value")
pyplot.title("HBV Graph")
axes = pyplot.gca()
axes.set_xlim([0, 35000000])
axes.set_ylim([0, 40000])
pyplot.show()


