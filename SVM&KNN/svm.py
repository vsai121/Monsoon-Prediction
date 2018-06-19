from sklearn import svm
import loader as l

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

X_train,y_train,X_test,y_test,_ = l.process()


clf = svm.SVC(kernel="linear", decision_function_shape='ovo',class_weight={0:1,1:0.5,2:1.5},C = 1, gamma = 0.001)
clf.fit(X_train, y_train)


pred = clf.predict(X_test)


print("Accuracy" , accuracy_score(y_test,pred))

print("Confusion matrix" , confusion_matrix(y_test,pred))

print("F1 score" , f1_score(y_test,pred, average='weighted'))


fig = plt.figure()
fig.subplots_adjust(bottom=0.2)
ax1 = fig.add_subplot(111)

line1 = ax1.plot(pred,'bo-',label='list 1')
line2 = ax1.plot(y_test,'go-',label='list 2')

plt.show()
