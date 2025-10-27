import pandas as pd
import math
import matplotlib.pyplot as plt

trainData = pd.read_csv("C:\\Users\\xxxsn\\OneDrive\\Desktop\\project3\\train.csv", index_col = "Id")

nTrainRecords = len(trainData)  # number of records in training data
meanTrainPrice = trainData["Price"].mean()   # mean of training date prices.
maxTrainPrice = trainData["Price"].max()    # max price in trainingData
minTrainPrice = trainData["Price"].min()    # min price in trainingData
trainStanDev = trainData["Price"].std() # standard deviation of house prices
print("Number of Records:", nTrainRecords, "Mean:", meanTrainPrice, "Minal Price ", minTrainPrice, "Max Price", maxTrainPrice)

plt.hist(trainData["Price"])
plt.xlabel("Price Value")
plt.ylabel("Frequency")
plt.show()

plt.scatter(trainData["GrLivArea"], trainData["Price"])
plt.xlabel("GrLivArea")
plt.ylabel("Price")
plt.show()
plt.scatter(trainData["BedroomAbvGr"], trainData["Price"])
plt.xlabel("BedroomAbvGr")
plt.ylabel("Price")
plt.show()
plt.scatter(trainData["TotalBsmtSF"], trainData["Price"])
plt.xlabel("TotalBsmtSF")
plt.ylabel("Price")
plt.show()
plt.scatter(trainData["FullBath"], trainData["Price"])
plt.xlabel("FullBath")
plt.ylabel("Price")
plt.show()
# based on the data shown here, I might place a low weight or even no weight entireley on BedroomAbvGr to avoid overfitting
# since it shows little to no correlation with price.

"""takes the id of a house (house) and a list of weights (w) and returns estimated price of that SINGLE house.
Used only for testing """
def predSingle(house, w):
    estPrice = 0
    for i in range (len(trainData.loc[house])):
        estPrice += trainData.loc[house][i]*w[i]
    return estPrice

"""takes a pandas dataframe, (data), a list of house ids (houses), and a list of weights (w) and returns a list of predicted prices 
in order."""
def pred(data, houses, w):
    return data.loc[houses].iloc[:, :-1].values @ weights[:-1]  # Compute predictions
 
weights = [0]*26

"""takes a pandas dataframe (data), and list of house Ids (houses) and calculates the MSE of the group of said houses. 
Calls pred function."""
def loss(data, houses):
    prices = data.loc[houses, "Price"].values  # Extract true prices
    predictions = pred(data, houses, weights)
    errors = predictions - prices  # Compute residuals
    return (errors @ errors) / len(houses)  # Compute MSE directly

def printWeights():
    print(weights)

"""gradient descent algorithm for correcting weights. takes a pandas dataframe and returns a correction for weights"""
def gradient(X, batch_size=32):
    sample = X.sample(batch_size)
    predictions = sample.iloc[:, :-1] @ weights[:-1]
    errors = predictions - sample["Price"]
    return (2 / batch_size) * (sample.iloc[:, :-1].T @ errors)

"""Both updates and returns weights. takes an alpha value (a), and list of corrections (grade)"""
def update(a, grade):
    for i in range (len(grade)):
        weights[i] -= a*grade[i]
    return weights

if __name__=="__main__":
    alpha = 0.2
    lastLoss = loss(trainData, trainData.index)
    update(alpha, gradient(trainData))
    iterations = 1
    print("Abc", lastLoss)
    print("def", loss(trainData, trainData.index))
    while lastLoss > loss(trainData, trainData.index):
        print("iteration #", iterations)
        update(alpha, gradient(trainData))
        lastLoss = loss(trainData, trainData.index)
        iterations += 1
    # it seems that when alpha = 0.2, error grows exponentialy due to overshoot.
    
    alpha2 = 10**-11
    alpha3 = 10**-12
    iterations1 = []
    MSE1 = []
    weights = [0]*26
    rememberWeights1 = []
    for i in range(200000):
        print("iteration #", i, "MSE", loss(trainData, trainData.index))
        update(alpha2, gradient(trainData))
        iterations1.append(i)
        MSE1.append(loss(trainData, trainData.index))
    rememberWeights1 = weights

    weights = [0]*26
    print(printWeights())
    iterations2 = []
    MSE2 = []
    rememberWeights2 = []
    for i in range(200000):
        print("iteration #", i, "MSE", loss(trainData, trainData.index))
        update(alpha3, gradient(trainData))
        iterations2.append(i)
        MSE2.append(loss(trainData, trainData.index))
        #print(MSE2)
    remmeberWeights2 = weights

    print("weights 1:", rememberWeights1)
    print("weights 2", remmeberWeights2)

    plt.plot(iterations1, MSE1, label="Alpha = 10^-11")  # Add label
    #print(MSE2) 
    plt.plot(iterations2, MSE2, label="Alpha = 10^-12")  # Add label
    plt.xlabel("Iterations")  # Add X-axis label
    plt.ylabel("Mean Squared Error (MSE)")  # Add Y-axis label
    plt.title("Loss Over Iterations")  # Add plot title
    plt.legend()  # Display the legend
    plt.show()

    testData = pd.read_csv("C:\\Users\\xxxsn\\OneDrive\\Desktop\\project3\\test.csv", index_col = "Id")
    print("ABC", pred(testData, testData.index, weights))
    print("test MSE: ", loss(testData, testData.index))