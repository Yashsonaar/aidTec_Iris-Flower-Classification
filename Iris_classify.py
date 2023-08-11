import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

a = pd.read_csv("IRIS.csv")
print(a)
print(a.isnull().sum())
print(a.species.unique())
print(a.info)
a = a.fillna(0)
train = a.iloc[:, 0:4]
test = a.species
X_train, Xtest, y_train, ytest = train_test_split(train, test, test_size=0.25)
plt.hist(train.sepal_length, bins=20, color="orange")
# plt.scatter(train.sepal_length, train.petal_length)
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.show()
classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(Xtest)
print(accuracy_score(ytest, y_pred))
# b = pd.read_csv("Book.csv")
new_data = [[5.1,3.7,1.4,0.3], [6,2.2,4,1], [6.4,2.7,5.3,1.9]]
ans = classifier.predict(new_data)
print(ans)
