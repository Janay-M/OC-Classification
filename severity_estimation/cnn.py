import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Convolution1D, MaxPooling1D
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from statistics import mean, stdev
from sklearn.metrics import accuracy_score

#CREATE AN EMPTY LIST TO STORE THE ACCURACY OF EACH FOLD IN
results = []

#CREATE A LOOP FOR THE 3-FOLD CROSS VALIDATION
for f in range(1, 4):
    features_train = np.load("/PATH_WHERE_FOLDS_FEATURES_ARE_STORED/fold{}/train.npy".format(f))
    features_test = np.load("/PATH_WHERE_FOLDS_FEATURES_ARE_STORED/fold{}/test.npy".format(f))
    labels_train = np.load("/PATH_WHERE_FOLDS_LABELS_ARE_STORED/fold{}/train.npy".format(f))
    labels_test = np.load("/PATH_WHERE_FOLDS_LABELS_ARE_STORED/fold{}/test.npy".format(f))

#CREATE THE MODEL, FIT, PREDICT AND GET ACCURACY RESULTS
#INPUT SHAPE DEPENDS ON THE FEATURE: (2050,1) FOR LTAS; (9380,1) FOR MFCC)
    model = Sequential()
    model.add(Convolution1D(32, kernel_size = 3, activation = 'relu', input_shape = ("INPUT_SHAPE")))
    model.add(Convolution1D(64, 3, activation = 'relu'))
    model.add(Convolution1D(128, 2, activation = 'relu'))
    model.add(Convolution1D(256, 2, activation = 'relu'))
    model.add(MaxPooling1D(pool_size = 2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(units=128, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(10, activation = 'softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
    model.fit(features_train, labels_train,
        callbacks=EarlyStopping(monitor="val_loss", verbose=1, mode='auto', patience=1, restore_best_weights=True),
        batch_size = 64,
        epochs = 7,
        verbose = 1,
        validation_data = (features_test, labels_test)
        )

    loss, accuracy = model.evaluate(features_test, labels_test, verbose=0)
    results.append(accuracy)

print('List of possible accuracies:', results)
print('\nMaximum accuracy that can be obtained from this model is:', max(results)*100, '%')
print('\nMinimum accuracy:', min(results)*100, '%')
print('\nOverall accuracy:', mean(results)*100, '%')
print('\nStandard Deviation is:', stdev(results))
