import tensorflow
import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# Read
data = pd.read_csv("student-mat.csv", sep=";")

# Trim
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# Predict value
predict = "G3"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)


def retrain_model():
    best = 0
    for _ in range(100):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

        # Best fit line for X,y (train model)
        linear = linear_model.LinearRegression()

        linear.fit(x_train, y_train)

        # Test data
        acc = linear.score(x_test, y_test)
        print("Accuracy: \n", acc)

        if acc > best:
            best = acc
            # Save pickle file and model
            with open("studentmodel.pickle", "wb") as f:
                pickle.dump(linear, f)


'''retrain_model()'''

# Read pickle file
pickle_in = open("studentmodel.pickle", "rb")

# Load model into our linear variable
linear = pickle.load(pickle_in)

# Test model (our acc) on data
print("Coefficient: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

# Use model to predict
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(
        "\n====================================="
        "\nPredicated Final Grade: ", predictions[x],
        "\nGrade History: ", x_test[x],
        "\nActual Final Grade: ", y_test[x],
        "\nAcc: ", "acc", ", Coefficient: ", linear.coef_, ", Intercept: ", linear.intercept_)

# Show grid/plot
p = 'G1'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
