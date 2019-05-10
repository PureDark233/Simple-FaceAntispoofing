import numpy as np
import json
from sklearn import model_selection as mo
from sklearn import svm
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import skimage

file = "data.json"


def spoof_type(s):
    spoof = {'ClientRawface': 0, 'ImposterRawface': 1}
    return spoof[s]


x_train = []
x_test = []
y_train = []
y_test = []
y0 = np.array([0])
y1 = np.array([1])
with open(file, 'r')as f:
    list = json.load(f)
    data = np.array(list)
for i in range(len(data)):
    # print(data[i])
    x, _ = (np.split(data[i], (1536,), axis=0))

    if data[i][-1] == 'train_all/real':
        x_train.append(x)
        y_train.append(0)
    if data[i][-1] == 'test_all/real':
        x_test.append(x)
        y_test.append(0)
    if data[i][-1] == 'train_all/attack':
        x_train.append(x)
        y_train.append(1)
    if data[i][-1] == 'test_all/attack':
        x_test.append(x)
        y_test.append(1)
x_train = np.array((x_train))
x_test = np.array((x_test))
y_train = np.array((y_train))
y_test = np.array((y_test))
# x_train,x_test,y_train,y_test=mo.train_test_split(x,y,random_state=1,train_size=0.6)
classifier = svm.SVC(C=0.9, kernel='rbf', gamma=30, decision_function_shape='ovo')  # C=0.8, gamma=20
classifier.fit(x_train, y_train.ravel())
print(classifier.score(x_train, y_train))

y_hat = classifier.predict(x_train)
print(classification_report(y_train, y_hat, digits=3))

y_hat = classifier.predict(x_test)
print(classification_report(y_test, y_hat, digits=3))
