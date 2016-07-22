import tensorflow as tf

import numpy as np
from numpy import count_nonzero

# Data sets
IRIS_TRAINING = "tfidfvectorsnew1.csv"
IRIS_TEST = "tfidfvectorsnew2.csv"

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TRAINING, target_dtype=np.int64)
test_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TEST, target_dtype=np.int64)

x_train, x_test, y_train, y_test = training_set.data, test_set.data, \
  training_set.target, test_set.target
pos=0
for a in y_train:
    if a=='1':
        pos=pos+1
print(pos)
pos=0
for a in y_test:
    if a=='1':
        pos=pos+1
print(pos)
avgauc=0
avgf1=0
ba=10
# Build 3 layer DNN with 10, 20, 10 units respectively.
for i in range(ba):
    classifier = tf.contrib.learn.DNNClassifier(hidden_units=[20,20,20,50, 50,50,50, 100], n_classes=2)

    # Fit model.
    classifier.fit(x=x_train, y=y_train, steps=1000)
    y = classifier.predict(x_test)
    print(count_nonzero(y))
    print (str(y))
    print(y_test)
    tp=0
    tn=0
    fn=0
    fp=0
    for c in range(len(y_test)):
        if y_test[c]=='1' and y[c]==1:
            tp+=1
        if y_test[c]=='0' and y[c]==1:
            fp+=1
        if y_test[c]=='1' and y[c]==0:
            fn+=1
        if y_test[c]=='0' and y[c]==0:
            tn+=1
    # Evaluate accuracy.
    if tp==0:
        precision=0
        recall=0
        f1=0
    else:
        precision= tp/(tp+fp)
        recall=tp/(tp+fn)
        f1=2*(precision*recall)/(precision+recall)
    avgf1+=f1
    accuracy_score = classifier.evaluate(x=x_test, y=y_test)['eval_auc']
    avgauc+=accuracy_score
    print("Accuracy: "+str(accuracy_score))
    print("F1: "+str(f1))
avgauc=avgauc/ba
avgf1=avgf1/ba
print("Average auc: "+ str(avgauc))
print("Average f1: "+str(avgf1))