import tensorflow as tf

import numpy as np
from numpy import count_nonzero
from tensorflow.contrib.metrics.python import metrics

#Data sets
#returns False if the index is outside the array bounds or if the array value at that index isn’t searchterm
def arraysearch(a, searchterm, sindex):
    if sindex>=len(a):
        return False
    elif a[sindex]==searchterm:
        return True
    else:
        return False
#returns the lexicographically next term in the sequence of possible nodes and layers
def testincrements(a, start, end, increment):
    c=0
    while arraysearch(a, end, c) and c<len(a):
        a[c]=start
        c+=1
    if c==len(a):
        a.append(start)
        return a
    else:
        a[c]=a[c]+increment
        return a

IRIS_TRAINING = "tfidfvectorsnew1.csv"
IRIS_TEST = "tfidfvectorsnew2.csv"

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TRAINING, target_dtype=np.int64, has_header=False)
test_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TEST, target_dtype=np.int64, has_header=False)

x_train, x_test, y_train, y_test = training_set.data, test_set.data, \
  training_set.target, test_set.target
pos=0
#get and print the number of positive examples(parse check)
for a in y_train:
    if a=='1':
        pos=pos+1
print(pos)
pos=0
for a in y_test:
    if a=='1':
        pos=pos+1
print(pos)
#initializing relevant variables(ba=number of loops per avg f1, all else self explanatory)
avgauc=0
avgf1=0
ba=20
optimizerarraystart=[20, 20, 20]
optimizerarraycurrent=[20, 20, 20];
maxoptimizerarray=optimizerarraystart;
maxavgf1=0
# Build multilayer DNN with current array of layers and nodes, set <x to 6 on server.
while len(optimizerarraycurrent)<6:
    for i in range(ba):
        #create model
        classifier = tf.contrib.learn.DNNClassifier(hidden_units=optimizerarraycurrent, n_classes=2)
   

        # Fit model.
        classifier.fit(x=x_train, y=y_train, steps=1000)
        y = classifier.predict(x_test)
        # Evaluate accuracy, first by printing predicted and actual labelings for the test set and then by calculating f1
        print(count_nonzero(y))
        print (str(y))
        print(y_test)
        #actually getting the f1 score
        tp=float(0.0)
        tn=float(0.0)
        fn=float(0.0)
        fp=float(0.0)
        for c in range(len(y_test)):
            if y_test[c]=='1' and y[c]==1:
                tp+=1
            if y_test[c]=='0' and y[c]==1:
                fp+=1
            if y_test[c]=='1' and y[c]==0:
                fn+=1
            if y_test[c]=='0' and y[c]==0:
                tn+=1
        
        if tp+fp==0 or tp+fn==0:
            precision=0
            recall=0
            f1=0
        else:
            precision= tp/(tp+fp)
            recall=tp/(tp+fn)
            f1=2*(precision*recall)/(precision+recall)
        avgf1+=f1
        #here the program calculates auc
        accuracy_score = classifier.evaluate(x=x_test, y=y_test)['eval_auc']
        avgauc+=accuracy_score
        print("Accuracy: "+str(accuracy_score))
        print("F1: "+str(f1))
        print("Precision : " + str(precision))
        print("Recall : " + str(recall))
    #at this point the program calculates the average f1 and auc for this setup of layers and nodes
    avgauc=avgauc/ba
    avgf1=avgf1/ba
    print(metrics);
    print("Average auc: "+ str(avgauc))
    print("Average f1: "+str(avgf1))
    #here's the maximization code
    if avgf1>maxavgf1:
        maxoptimizerarray=optimizerarraycurrent
        maxavgf1=avgf1
    #change these values as needed(20, 50, 15) are suggested for last 3 params
    #here we get the lexicographically next setup of nodes and layers
    optimizerarraycurrent=testincrements(optimizerarraycurrent, 20, 50, 15) 
#after the program has tested every setup of nodes and layers given it's starting params, we print out the best setup and its average f1
print(str(maxoptimizerarray))
print(str(maxavgf1))
