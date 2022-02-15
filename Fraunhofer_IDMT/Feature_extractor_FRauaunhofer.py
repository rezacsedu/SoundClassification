import librosa
import os
import pandas as pd
import glob
import numpy as np
import sys

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

#parent_dir = 
#number_epoch = 2

def get_class_names(path):  # class names are subdirectory names in Samples/ directory
    class_names = os.listdir(path)[0]
    return class_names

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size // 2)

def extract_features(parent_dir,sub_dirs,file_ext="*.wav",bands = 20, frames = 41):
    window_size = 512 * (frames - 1)
    mfccs = []
    labels = []
    for l, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            sound_clip,s = librosa.load(fn)

            if (sub_dir == 'Overdrive'):
                label = 1
            elif(sub_dir == 'Phaser'):
                label = 2
            elif(sub_dir == 'Reverb'):
                label = 3
            elif(sub_dir == 'SlapbackDelay'):
                label = 4
            elif(sub_dir == 'Tremolo'):
                label = 5
            else:
                label = 6

            #label = fn.split('fold')[1].split('-')[1]
            #label = get_class_names("Samples/")   # get the names of the subdirectories
            #nb_classes = len(label)
            print("class_names = ",label)

            for (start,end) in windows(sound_clip,window_size):
                if(len(sound_clip[start:end]) == window_size):
                    signal = sound_clip[start:end]
                    mfcc = librosa.feature.mfcc(y=signal, sr=s, n_mfcc = bands).T.flatten()[:, np.newaxis].T
                    mfccs.append(mfcc)
                    labels.append(label)         
    features = np.asarray(mfccs).reshape(len(mfccs),bands,frames)
    return np.array(features), np.array(labels,dtype = np.int)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

parent_dir = '/home/asif/Downloads/Fraunhofer_IDMT/Samples/'
sub_dirs = ['Overdrive','Phaser','Reverb','SlapbackDelay', 'Tremolo','Vibrato']
file_ext = "*.wav"

# use this to process the audio files into numpy arrays
def save_folds(data_dir):
    for k in range(len(sub_dirs)):
        fold_name = 'fold' + str(k)
        print("\nSaving " + fold_name)
        features, labels = extract_features(parent_dir, sub_dirs)
        print(labels)
        #labels = one_hot_encode(labels)
        
        print("Features of", fold_name , " = ", features.shape)
        print("Labels of", fold_name , " = ", labels.shape)
        
        feature_file = os.path.join(data_dir, fold_name + '_x.npy')
        labels_file = os.path.join(data_dir, fold_name + '_y.npy')
        np.save(feature_file, features)
        print("Saved " + feature_file)
        np.save(labels_file, labels)
        print("Saved " + labels_file)

def assure_path_exists(path):
    mydir = os.path.join(os.getcwd(), path)
    if not os.path.exists(mydir):
        os.makedirs(mydir)

if __name__ == "__main__":
    save_folds('/home/asif/Downloads/Fraunhofer_IDMT/Features/')
