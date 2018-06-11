from sklearn import svm
import loader as l

from sklearn.metrics import *

X_train,y_train,X_validation,y_validation,X_test,y_test = l.process()


clf = svm.SVC(kernel="rbf", decision_function_shape='ovo', class_weight='balanced')
clf.fit(X_train, y_train)


predicted = clf.predict(X_test)

cnt=0
for i in range(len(predicted)):
    print(predicted[i]),
    print(y_test[i])

    if(predicted[i]==y_test[i] and y_test[i]!=1):
        cnt+=1

print(float(cnt)/len(predicted))
print accuracy_score(y_test, predicted)
