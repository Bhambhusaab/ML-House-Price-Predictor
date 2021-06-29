from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
iris = datasets.load_iris()
print(iris.keys())
print(iris['data'].shape)
print(iris['data'])
print(iris['target'])
print(iris['DESCR'])
x = iris['data'][:, 3:]
print(x)
y = (iris['target'] == 2).astype(np.int)
print(y)
clf = LogisticRegression()
# clf = KNeighborsClassifier()
clf.fit(x, y)

ex = clf.predict(([[67]]))
print(ex)
x_new = np.linspace(0, 3, 1000).reshape(-1, 1)
print(x_new)
y_prob = clf.predict_proba(x_new)
print(y_prob)
plt.plot(x_new, y_prob[:, 1], "g-", label="Virginica")
plt.show()
