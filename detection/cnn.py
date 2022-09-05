import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution1D, MaxPooling1D
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from statistics import mean, stdev

#CREATE AN EMPTY LIST TO STORE THE ACCURACY OF EACH FOLD IN
results = []

#CREATE A LOOP FOR THE 5-FOLD CROSS VALIDATION
for f in range(1, 6):
    features_train = np.load("/PATH_WHERE_FOLDS_FEATURES_ARE_STORED/fold{}/train.npy".format(f))
    features_test = np.load("/PATH_WHERE_FOLDS_FEATURES_ARE_STORED/fold{}/test.npy".format(f))
    labels_train = np.load("/PATH_WHERE_FOLDS_LABELS_ARE_STORED/fold{}/train.npy".format(f))
    labels_test = np.load("/PATH_WHERE_FOLDS_LABELS_ARE_STORED/fold{}/test.npy".format(f))

#THIS STEP IS NECESSARY IF YOUR LABELS (1, 0) HAVE BEEN SAVED AS STRINGS ('1', '0')
#IF THIS IS THE CASE, USE THE NEW VARIABLE NAMES IN THE MODEL
    labels_train1 = tf.strings.to_number(labels_train, out_type=tf.float32)
    labels_test1 = tf.strings.to_number(labels_test, out_type=tf.float32)

#CREATE THE MODEL, FIT, PREDICT AND GET ACCURACY RESULTS
#INPUT SHAPE DEPENDS ON THE FEATURE: (2050,1) FOR LTAS; (9380,1) FOR MFCC)
    model = Sequential()
    model.add(Convolution1D(32, kernel_size = 3, activation = 'relu', input_shape = (2050,1)))
    model.add(Convolution1D(64, 3, activation = 'relu'))
    model.add(Convolution1D(128, 2, activation = 'relu'))
    model.add(Convolution1D(256, 2, activation = 'relu'))
    model.add(MaxPooling1D(pool_size = 2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(units=128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = 'softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
    model.fit(features_train, labels_train1,
        callbacks=EarlyStopping(monitor="val_loss", verbose=1, mode='auto', patience=3, restore_best_weights=True),
        batch_size = 64,
        epochs = 10,
        verbose = 1,
        validation_data = (features_test, labels_test1)
        )

    loss, accuracy = model.evaluate(features_test, labels_test1, verbose=0)
    results.append(accuracy)

print('List of possible accuracies:', results)
print('\nMaximum accuracy that can be obtained from this model is:', max(results)*100, '%')
print('\nMinimum accuracy:', min(results)*100, '%')
print('\nOverall accuracy:', mean(results)*100, '%')
print('\nStandard Deviation is:', stdev(results))

