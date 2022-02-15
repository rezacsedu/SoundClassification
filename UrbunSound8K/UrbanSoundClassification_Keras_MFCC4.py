import librosa
import os
import pandas as pd
import glob
import numpy as np
import sys
from time import time

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import LSTM, Dense, Dropout, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from keras.utils import np_utils

from keras import backend as K
K.set_image_dim_ordering('tf')

data_dir = 'features/'
audiofile_base = '/audio/'

# this is used to load the folds incrementally
def load_folds(folds):
    subsequent_fold = False
    for k in range(len(folds)):
        fold_name = 'fold' + str(folds[k])
        feature_file = os.path.join(data_dir, fold_name + '_x.npy')
        labels_file = os.path.join(data_dir, fold_name + '_y.npy')
        loaded_features = np.load(feature_file)
        loaded_labels = np.load(labels_file)
        print (fold_name, "features: ", loaded_features.shape)

        if subsequent_fold:
            features = np.concatenate((features, loaded_features))
            labels = np.concatenate((labels, loaded_labels))
        else:
            features = loaded_features
            labels = loaded_labels
            subsequent_fold = True
        
    return features, labels

def prepare_test_train_valid():
    #train = pd.read_csv(str(base_path + metadata_base))
    #print(train.Class.value_counts(normalize=True))  # distribution of data

    features, labels = load_folds([1,2, 3, 4, 5, 6, 7, 8, 9, 10])
    # load fold1 for testing
    #train_x, train_y = load_folds([1,2,3,4,5,6])

    # load fold2 for validation
    #valid_x, valid_y = load_folds([9])
    
    # load fold3 for testing
    #test_x, test_y = load_folds([10])

    # Train-test split 
    train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.40, random_state=100)
    # train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
    test_x, valid_x, test_y, valid_y = train_test_split(train_x, train_y, test_size=0.50, random_state=100)
    return train_x, test_x, train_y, test_y, valid_x, valid_y

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

# Extract feature
train_x, test_x, train_y, test_y, valid_x, valid_y = prepare_test_train_valid()

train_x_cnn = train_x.reshape(train_x.shape[0], 20, 41, 1).astype('float32')
test_x_cnn = test_x.reshape(test_x.shape[0], 20, 41, 1).astype('float32')
valid_x_cnn = valid_x.reshape(valid_x.shape[0], 20, 41, 1).astype('float32')

bands = 20
frames = 41

train_x_dnn = train_x.reshape(train_x.shape[0], bands*frames).astype('float32')
test_x_dnn = test_x.reshape(test_x.shape[0], bands*frames).astype('float32')
valid_x_dnn = valid_x.reshape(valid_x.shape[0], bands*frames).astype('float32')

print('X_train shape:', train_x.shape)
num_classes = 10

def build_DNN():#Wrong   
    model = Sequential()
    model.add(Dense(64, input_dim=train_x_dnn.shape[1], kernel_initializer="uniform")) # X_train.shape[1] == 15 here
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(64, kernel_initializer="uniform"))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(train_y.shape[1], kernel_initializer="uniform")) # y_train.shape[1] == 2 here
    model.add(Activation('softmax'))
    return model

def build_LSTM(): #OK
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    # returns a sequence of vectors of dimension 256
    model.add(LSTM(256, return_sequences=True, input_shape=(bands, frames)))  
    model.add(Dropout(0.2))
    # return a single vector of dimension 128
    model.add(LSTM(128))  
    model.add(Dropout(0.2))
    # apply softmax to output
    model.add(Dense(num_classes, activation='softmax'))
    return model

def build_CNN(): #OK
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(bands, frames, 1), activation='relu'))
    model.add(MaxPooling2D(data_format="channels_last", pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax')) 
    return model

def build_DeepCNN():
    f_size = 5
    pool_size=(4, 2)
    model = Sequential()   
    num_channels = 2 

    model.add(Conv2D(32, (f_size, f_size), padding='same', input_shape=(bands, frames, 1), data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (f_size, f_size)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (f_size, f_size), padding='same', input_shape=(bands, frames, 1), data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (f_size, f_size)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, W_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, W_regularizer=l2(0.001)))
    model.add(Activation('softmax'))
    return model 

def model_train_evaluate(model, number_epoch):   
    sgd = SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)

    # a stopping function should the validation loss stop improving
    earlystop = EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')

    if model in ['DNN']: 
        dnn_model = build_DNN()
        dnn_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)
        tensorboardDNN = TensorBoard(log_dir="DNN_logs/{}".format(time()))
        dnn_model.fit(train_x_dnn, train_y, validation_data=(valid_x_dnn, valid_y), callbacks=[tensorboardDNN], batch_size=128, epochs=int(number_epoch))
        print(dnn_model.summary())
        
        # serialize model to JSON
        model_json = dnn_model.to_json()
        with open("TrainedModels/model_RNN.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        dnn_model.save_weights("TrainedModels/model_RNN.h5")
        print("Saved model to disk")
        
        y_prob = dnn_model.predict(test_x_dnn) 
        y_pred = y_prob.argmax(axis=-1)
        y_true = np.argmax(test_y, 1)

        roc = roc_auc_score(test_y, y_prob)
        print ("ROC:",  round(roc,3))

        # evaluate the model
        score, accuracy = dnn_model.evaluate(test_x_dnn, test_y, batch_size=32)
        print("\nAccuracy = {:.2f}".format(accuracy))

        # the F-score gives a similiar value to the accuracy score, but useful for cross-checking
        p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average='micro')
        print ("F-Score:", round(f,2))
        print ("Precision:", round(p,2))
        print ("Recall:", round(r,2))
        print ("F-Score:", round(f,2))
        '''
        # load json and create model
        json_file = open('TrainedModels/model_DNN.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("TrainedModels/model_DNN.h5")
        print("Model restored from disk")

        # evaluate loaded model on test data
        loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        score = loaded_model.evaluate(test_x_dnn, test_y, verbose=0)
        print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
        '''
        import gc; gc.collect()

    if model in ['RNN']: 
        rnn_model = build_LSTM() #OK
        rnn_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)
        tensorboardRNN = TensorBoard(log_dir="RNN_logs/{}".format(time()))
        rnn_model.fit(train_x, train_y, validation_data=(valid_x, valid_y), callbacks=[tensorboardRNN], batch_size=128, epochs=int(number_epoch))
        print(rnn_model.summary())
        
        y_prob = rnn_model.predict(test_x) 
        y_pred = y_prob.argmax(axis=-1)
        y_true = np.argmax(test_y, 1)

        roc = roc_auc_score(test_y, y_prob)
        print ("ROC:",  round(roc,3))

        # evaluate the model
        score, accuracy = rnn_model.evaluate(test_x, test_y, batch_size=32)
        print("\nAccuracy = {:.2f}".format(accuracy))

        # the F-score gives a similiar value to the accuracy score, but useful for cross-checking
        p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average='micro')
        print ("F-Score:", round(f,2))
        print ("Precision:", round(p,2))
        print ("Recall:", round(r,2))
        print ("F-Score:", round(f,2))

    if model in ['CNN']:
        cnn_model = build_CNN()
        cnn_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)
        tensorboardCNN = TensorBoard(log_dir="CNN_logs/{}".format(time()))
        cnn_model.fit(train_x_cnn, train_y, validation_data=(valid_x_cnn, valid_y), callbacks=[tensorboardCNN], batch_size=128, epochs=int(number_epoch))
        print(cnn_model.summary())

        # serialize model to JSON
        model_json = dnn_model.to_json()
        with open("TrainedModels/model_CNN.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        dnn_model.save_weights("TrainedModels/model_CNN.h5")
        print("Saved model to disk")

        y_prob = cnn_model.predict(test_x_cnn) 
        y_pred = y_prob.argmax(axis=-1)
        y_true = np.argmax(test_y, 1)

        roc = roc_auc_score(test_y, y_prob)
        print ("\nROC:",  round(roc,3))

        # evaluate the model
        score, accuracy = cnn_model.evaluate(test_x_cnn, test_y, batch_size=32)
        print("\nAccuracy = {:.2f}".format(accuracy))

        # the F-score gives a similiar value to the accuracy score, but useful for cross-checking
        p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average='micro')
        print ("Precision:", round(p,2))
        print ("Recall:", round(r,2))
        print ("F-Score:", round(f,2))

    if model in ['DeepCNN']:
        deep_cnn_model = build_DeepCNN()
        deep_cnn_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)
        tensorboardDeepDNN = TensorBoard(log_dir="DeepCNN_logs/{}".format(time()))
        deep_cnn_model.fit(train_x_cnn, train_y, validation_data=(valid_x_cnn, valid_y), callbacks=[tensorboardDeepDNN], batch_size=128, epochs=int(number_epoch))
        print(deep_cnn_model.summary())

        # serialize model to JSON
        model_json = deep_cnn_model.to_json()
        with open("TrainedModels/model_DeepCNN.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        deep_cnn_model.save_weights("TrainedModels/model_DeepCNN.h5")
        print("Saved model to disk")

        y_prob = deep_cnn_model.predict(test_x_cnn) 
        y_pred = y_prob.argmax(axis=-1)
        y_true = np.argmax(test_y, 1)

        roc = roc_auc_score(test_y, y_prob)
        print ("\nROC:",  round(roc,3))

        # evaluate the model
        score, accuracy = deep_cnn_model.evaluate(test_x_cnn, test_y, batch_size=32)
        print("\nAccuracy = {:.2f}".format(accuracy))

        # the F-score gives a similiar value to the accuracy score, but useful for cross-checking
        p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average='micro')
        print ("Precision:", round(p,2))
        print ("Recall:", round(r,2))
        print ("F-Score:", round(f,2))

if __name__ == "__main__":
    model = sys.argv[1]
    num_epoch = sys.argv[2]
    model_train_evaluate(model, num_epoch)