import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("part4-classification/suv_data.csv")
data['Gender'].replace(['Male','Female'],[0,1],inplace=True)

x = data[["Age", "EstimatedSalary", "Gender"]].values
y = data["Purchased"].values

# Step 2: Standardize the data using StandardScaler, 
scaler = StandardScaler().fit(x)
x = scaler.transform(x)

# Step 4: Split the data into training and testing data

x_train, x_test, y_train, y_test = train_test_split(x, y)

model = linear_model.LogisticRegression().fit(x_train, y_train)


# Step 6: Create a LogsiticRegression object and fit the data

# Step 7: Print the score to see the accuracy of the model

# Step 8: Print out the actual ytest values and predicted y values
# based on the xtest data

print("Accuracy:", model.score(x_test, y_test))
print("Testing Results:")
print(y_test)
for index in range(len(x_test)):
    x = x_test[index]
    x = x.reshape(-1, 3)
    y_pred = int(model.predict(x))

    actual = y_test[index]
    print("predicted", y_pred, "actual", actual)

myPrediction = [[34,56000,1]]
scale = scaler.transform(myPrediction)
answer = model.predict(scale)
print(answer)