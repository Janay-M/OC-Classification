import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from statistics import stdev, mean

#CREATE AN EMPTY LIST TO STORE THE ACCURACY OF EACH FOLD IN
results = []

#CREATE A LOOP FOR THE 5-FOLD CROSS VALIDATION
for f in range(1, 6):
    features_train = np.load("/PATH_WHERE_FOLDS_FEATURES_ARE_STORED/fold{}/train.npy".format(f))
    features_test = np.load("/PATH_WHERE_FOLDS_FEATURES_ARE_STORED/fold{}/test.npy".format(f))
    labels_train = np.load("/PATH_WHERE_FOLDS_LABELS_ARE_STORED/fold{}/train.npy".format(f))
    labels_test = np.load("/PATH_WHERE_FOLDS_LABELS_ARE_STORED/fold{}/test.npy".format(f))

#CREATE THE MODEL, FIT, PREDICT AND GET ACCURACY RESULTS
    svm = make_pipeline(StandardScaler(), SVC(gamma='auto',C=0.009, verbose=True, random_state=42, class_weight=None, shrinking=False, kernel='poly',max_iter=10000))
    svm.fit(features_train, labels_train)
    prediction = svm.predict(features_test)
    accuracy = accuracy_score(labels_test, prediction)
    results.append(accuracy)

print('List of possible accuracies:', results)
print('\nMaximum accuracy that can be obtained from this model is:', max(results)*100, '%')
print('\nMinimum accuracy:', min(results)*100, '%')
print('\nOverall accuracy:', mean(results)*100, '%')
print('\nStandard Deviation is:', stdev(results))
