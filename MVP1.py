import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
import sys
import csv

def refreshdata():
    global dataset
    dataset = pd.read_csv("PYTHONDATA.csv", sep=",") # reads in the data set
    global predict
    predict = "HBVDNACopies"
    global feature
    feature = "HBsAg"
    global x
    global y
    x = np.array(dataset.drop([predict], 1))  # separating the Feature what we are using to predict by dropping the Label
    y = np.array(dataset[predict])  # separating the Label what we are trying to predict
    train()


def train():

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)  # splits data 10% testing 90% training

    bestaccuracy = 0.9999999999999941 # current best accuracy of model
    for _ in range(100):  # for loop will repeat regression 100 times only saving if new model is more accurate than previous model this is good practice

        lineartrain = linear_model.LinearRegression()  # trains the regression model

        lineartrain.fit(x_train, y_train)
        acc = lineartrain.score(x_test, y_test)  #gets the accuracy of the current model
        if acc > bestaccuracy:
            bestaccuracy = acc # replaces current best accuracy with new best accuracy
            print('New Coefficient: ', lineartrain.coef_)  # These are each slope value
            print('New Intercept: ', lineartrain.intercept_)  # This is the intercept
            print('New Accuracy: ', bestaccuracy)
            with open("LinearRegressionModel.pickle", "wb") as f:
                pickle.dump(lineartrain, f)  # saves a pickle file of the model
    """"" TESTING CODE
    #predictions = linear.predict(x_test)
    #for x in range(len(predictions)):
        #print("Calculated HBV Value =", predictions[x], ". HBsAg Value =", x_test[x], ". Actual HBV Value =",y_test[x])  # Where the variables are in order - Actual HBV Value """

def calculation(HBsAg):
    pickle_in = open("LinearRegressionModel.pickle", "rb")
    linear = pickle.load(pickle_in)  # loads in the model and loads it into the script
    calculatedoutput = linear.intercept_ + (linear.coef_ * HBsAg)
    return calculatedoutput

def prediction():

    print("Please input your value for HBsAg or press Q to return to menu")
    try:
        userInput = float(input())
        if type(userInput) == int or type(userInput) == float:
            result = calculation(userInput)
            print("The calculated HBV DNA Copy is =", )
            print("Would you like to contribute this record to the dataset? (Y/N)")
            contribute = input()
            if contribute == "y" or contribute == "Y" or contribute == "yes" or contribute == "Yes":
                print("Awesome!")
                adddata(userInput, result)
                train()
                menu()
            else:
                print("No problem!")
                menu()
        elif userInput == "Q" or userInput == "q":
            menu()
        else:
            print("You have made an error!")
            prediction()
    except ValueError:
        print("Not a number! Try again.")
        prediction()



def adddata(predictioninput, calculatedoutput):
    writer = csv.writer(open("PYTHONDATA.csv", 'a', newline=''))
    calced = calculatedoutput.item()
    print("HBsAg = ", predictioninput, " HBV = ", calced)
    writer.writerow([predictioninput, calced])
    refreshdata()
    return True

def groupprediction():
    print("NOTE: Your file must be in csv format made up of only ONE column of data consisting of HBsAg values with no column name! ")
    print("Please input the full name of your file eg. PYTHONDATA.csv or press Q to return to the menu: ")
    filename = input()
    try:
        newdataset = pd.read_csv(filename, sep=",")
        newwriter = csv.writer(open(filename, 'a', newline=''))
    except:
        print("This file could not be found! Please check the spelling and try again.") # validation for file name
        groupprediction()
    listOfColumnNames = list(newdataset)
    lengthofColumns = len(listOfColumnNames)

    if lengthofColumns != 1: # Input validation for data
        print("You have more than one column, please try again with only one column!")
        groupprediction()
    if listOfColumnNames[0] != 'HBsAg': #input validation for name of column
        print("Please name your column HBsAg, please try again!")
        groupprediction()
    newdataset["HBsAg"] = pd.to_numeric(newdataset["HBsAg"], errors='coerce')  # Input validation for CSV
    length = len(newdataset.index)
    listofcalculated = []
    i = 0
    while i != length:
        HBS = newdataset.at[i, 'HBsAg']
        newcalculatedoutput = calculation(HBS)
        listofcalculated.append(newcalculatedoutput)
        newwriter.writerow([HBS, newcalculatedoutput])
        print("The inputted HBsag is = ", HBS, " .The calculated HBV DNA Copy is =", newcalculatedoutput)
        i = i + 1

    newdataset['HBVDNACopies'] = listofcalculated
    newdataset.to_csv(filename, sep=',', index=False)
    print("Your calculated has been uploaded to your file!")
    print("Would you like to contribute your data to the dataset? (Y/N)")
    contribute = input()
    if contribute == "y" or contribute == "Y" or contribute == "yes" or contribute == "Yes":
        print("Awesome!")
        a = 0
        while a != length:
            HBsAg = newdataset.at[a, 'HBsAg']
            HBV = newdataset.at[a, 'HBVDNACopies']
            adddata(HBsAg, HBV)
            a = a + 1

        train()
        menu()
    else:
        print("No problem!")
        menu()


def graph():
    refreshdata()
    style.use("ggplot")
    pyplot.scatter(dataset[feature], dataset[predict])
    pyplot.xlabel(feature)
    pyplot.ylabel(predict)
    pyplot.title("HBV Graph")
    axes = pyplot.gca()
    axes.set_xlim([0, 120000])
    axes.set_ylim([0, 110000000])
    pyplot.show()
    menu()

def menu():
    print("************MAIN MENU**************")
    print()

    choice = input("""
                      A: Calculate single HBV DNA Copy
                      B: Upload CSV data for prediction
                      C: Generate graph
                      Q: Quit/Log Out

                      Please enter your choice: """)

    if choice == "A" or choice =="a":
        prediction()
    elif choice == "B" or choice =="b":
        groupprediction()
    elif choice == "C" or choice =="c":
        graph()
    elif choice=="Q" or choice=="q":
        print("Thank you for using this application!")
        sys.exit
    else:
        print("You must only select either A,B,C, or D.")
        print("Please try again")
        menu()

refreshdata()
train()
menu()