import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import statistics

# problema 1
iris_data = pd.read_csv("01_iris.csv")
iris = pd.DataFrame(iris_data)

print(iris.dtypes)
print(iris.head(3))

# problema 2
print(iris.keys())
print(iris.shape)

# problema 3
print(f'Numarul de observatii = {len(iris_data)}')
print(f'Numarul de valori null = {iris_data.isnull().sum()}')
print(f'Numarul de valori NaN = {iris_data.isna().sum().sum()}')

# problema 4
print(iris_data.describe())

# problema 5
print(iris_data[iris_data['Species'] == 'Iris-setosa'])
print(iris_data[iris_data['Species'] == 'Iris-versicolor'])
print(iris_data[iris_data['Species'] == 'Iris-virginica'])

# problema 7
print(iris_data.drop('Id', axis=1).head(5))

# problema 8
#plt.figure(figsize=(13, 7))
sns.boxplot(x='Species', y='SepalLengthCm', data=iris_data)
plt.title('Sepalelor\'length distribution after species')
#plt.show()

#plt.figure(figsize=(8, 6))
sns.countplot(x='Species', data=iris_data)
plt.title('Number of observations per species')
#plt.show()

#plt.figure(figsize=(12, 10))
sns.pairplot(iris_data, hue='Species')
plt.suptitle('Pair of diagrams for the relation between the variables of species', y=1.02)
#plt.show()

# problema 9
#plt.figure(figsize=(8, 6))
sns.countplot(x='Species', data=iris_data)
plt.title('Frequencies of the three species')
plt.xlabel('Specie')
plt.ylabel('Frequency')
#plt.show()

# problema 10
counter = iris_data['Species'].value_counts()

#plt.figure(figsize=(11, 11))
plt.pie(counter, labels=counter.index, startangle=90, colors=['skyblue', 'lightgreen', 'lightcoral'])
plt.title('Frequency of the three species')
#plt.show()

#problema 11
#plt.figure(figsize=(11, 11))
axis_x = iris_data["SepalLengthCm"]
axis_y = iris_data["SepalWidthCm"]
plt.plot(axis_x, axis_y)
#plt.show()

#problema12
#plt.figure(figsize=(11, 11))
axis_x = iris_data["PetalLengthCm"]
axis_y = iris_data["PetalWidthCm"]
plt.plot(axis_x, axis_y)
#plt.show()

#problema13
#fara linii
plt.figure(figsize=(11, 11))
x = iris_data["SepalLengthCm"]
y = iris_data["SepalWidthCm"]
plt.plot(x, y, marker = "o", markersize=8, markerfacecolor="red")
x_next = iris_data["PetalLengthCm"]
y_next = iris_data["PetalWidthCm"]
plt.plot(x_next, y_next, marker = "o", markersize=8, markerfacecolor="green")
plt.show()

#problema14, 15
sns.jointplot(x = "SepalLengthCm", y = "SepalWidthCm", kind = "hex", data = iris_data)
#plt.show()

#problema16
#nu stiu

#problema17
#nu stiu

#problema18
#plt.figure(figsize = (11, 11))
heatm = sns.heatmap(data = iris_data)
#plt.show()

#problema19
#cred
X, y = iris_data.columns[:4], iris_data.keys()

#problema20
X_train, X_test, y_train, y_test = train_test_split(iris_data[["Id", "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]], iris_data["Species"], test_size=0.2, random_state=42)

#problema21
iris_data["Species"].replace(["Iris-setosa", "Iris-versicolor", "Iris-virginica"], [0, 1, 2], inplace=True)
X_train, X_test, y_train, y_test = train_test_split(iris_data[["Id", "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]], iris_data["Species"], test_size=0.2, random_state=42)

#problema22
X_train, X_test, y_train, y_test = train_test_split(iris_data[["Id", "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]], iris_data["Species"], test_size=0.3, random_state=42)

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)
#accuracy = accuracy_score(y_test, y_pred)

#problema23
X_train, X_test, y_train, y_test = train_test_split(iris_data[["Id", "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]], iris_data["Species"], test_size=0.2, random_state=42)
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

#problema 24, 25, 26
X_train, X_test, y_train, y_test = train_test_split(iris_data[["Id", "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]], iris_data["Species"], test_size=0.2, random_state=42)
for n in [2, 3, 4, 5, 6]:
    for leaf_s in [30, 40, 50, 60, 70, 80]:
        neigh = KNeighborsClassifier(n_neighbors=n, leaf_size=leaf_s)
        neigh.fit(X_train, y_train)
        y_pred = neigh.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

#trb facut grafic

#problema27
x = statistics.mean(iris_data[iris_data['Species'] == 'Iris-setosa'])
print(x)
x = statistics.mean(iris_data[iris_data['Species'] == 'Iris-versicolor'])
print(x)
x = statistics.mean(iris_data[iris_data['Species'] == 'Iris-virginica'])
print(x)
x = statistics.stdev(iris_data[iris_data['Species'] == 'Iris-setosa'])
print(x)
x = statistics.stdev(iris_data[iris_data['Species'] == 'Iris-versicolor'])
print(x)
x = statistics.stdev(iris_data[iris_data['Species'] == 'Iris-virginica'])
print(x)
x = np.percentile(iris_data[iris_data['Species'] == 'Iris-setosa'], 50)
print(x)
x = np.percentile(iris_data[iris_data['Species'] == 'Iris-versicolor'], 25)
print(x)
x = np.percentile(iris_data[iris_data['Species'] == 'Iris-virginica'], 75)
print(x)

