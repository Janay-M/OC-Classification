import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from statistics import stdev, mean

#CREATE AN EMPTY LIST TO STORE THE ACCURACY OF EACH FOLD IN
results = []

#CREATE A LOOP FOR THE 3-FOLD CROSS VALIDATION
for f in range(1, 4):
    features_train = np.load("/PATH_WHERE_FOLDS_FEATURES_ARE_STORED/fold{}/train.npy".format(f))        
    features_test = np.load("/PATH_WHERE_FOLDS_FEATURES_ARE_STORED/fold{}/test.npy".format(f))          
    labels_train = np.load("/PATH_WHERE_FOLDS_LABELS_ARE_STORED/fold{}/train.npy".format(f))         
    labels_test = np.load("/PATH_WHERE_FOLDS_LABELS_ARE_STORED/fold{}/test.npy".format(f))

#CREATE THE MODEL, FIT, PREDICT AND GET ACCURACY RESULTS
    mlp = MLPClassifier(random_state=42, batch_size='auto', verbose=True, max_iter=10000, activation='relu', solver='adam', learning_rate='adaptive')
    mlp.fit(features_train, labels_train)
    prediction = mlp.predict(features_test)
    accuracy = accuracy_score(labels_test, prediction)
    results.append(accuracy)

print('List of possible accuracies:', results)  
print('\nMaximum accuracy that can be obtained from this model is:', max(results)*100, '%') 
print('\nMinimum accuracy:', min(results)*100, '%')
print('\nOverall accuracy:', mean(results)*100, '%')
print('\nStandard Deviation is:', stdev(results))
