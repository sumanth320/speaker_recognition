#!/usr/bin/env python
# coding: utf-8




import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from scipy.io.wavfile import write

#Function to extract the features (of the Audio Clip) and to covert the matrix into a normalized form

def resize_mfcc(X,sample_rate):
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
    #print(len(mfccs))
    
    #tonnetz
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate)
    
    #melspectrogram
    mel = librosa.feature.melspectrogram(X, sr=sample_rate)
    
    #Flatness
    flat=librosa.feature.spectral_flatness(y=X)
    
    #Chroma_stft (Short Time Fourier Transform)
    chrome=librosa.feature.chroma_stft(y=X,sr=sample_rate)
    
    #Roll0off frequency. setting roll perentage to max or min
    rolloff=librosa.feature.spectral_rolloff(y=X,sr=sample_rate,roll_percent=0.85) #default roll_percent=0.85
    
    #Zero-crosing rate
    zero=librosa.feature.zero_crossing_rate(y=X)
    
    #Spectral Bandwidth
    bandwidth=librosa.feature.spectral_bandwidth(y=X,sr=sample_rate)
    
    #pitchs and magnitude
    pitchs,magnitude=librosa.piptrack(y=X,sr=sample_rate)
    
    #Vertically stacks all the feature arrays
    features = np.vstack([mfccs,mel,tonnetz,chrome,rolloff,zero,bandwidth,pitchs,magnitude])
    print("unstacked shape",features.shape)
    
    #To caculate the mean of the transposed feature array where axis = 0 indicates it is being worked along the Column
    stacked = np.mean(features.T,axis=0)
    
    #Returns the combined feature array
    return stacked


def soundrec():
    import sounddevice as sd
    #Sampling rate of 44.1KHz
    fs = 44100
    #Duration to be recorded
    seconds = 5
    print("Intitating Recording")
    sd.wait()
    #Informing user before the recording starts
    print("Starting")
    myrecording = sd.rec(int(seconds*fs), samplerate = fs, channels =2)
    sd.wait()
    #Writing the recorded audio in the specified location
    sd = write(r'.\Desktop\voice\Soundrec.wav', fs, myrecording)
    print("End of Recording")
    #Retuns the recording as part of function
    return sd


if __name__== "__main__":
    
    #Librosa function returns Audio Time Series and the Sampling Rate
    X1, sr1 = librosa.load(r'.\Desktop\voice\Speaker-275-4.wav', res_type='kaiser_fast')
    
    #Displaying the Audio Wave
    plt.figure(figsize=(12, 4))
    librosa.display.waveplot(X1, sr=sr1)
    
    #Passing both the above input to the Feature Extraction function
    combft = resize_mfcc(X1 , sr1)
    print(combft.shape)
    
    #Assigning the Target array and reshaping it to the size of feature array
    z = np.asarray(0)
    z2=np.resize(z,(2239,))
    
    #soundrec()
    
    #Test Audio File
    X2, sr2 = librosa.load(r'.\Desktop\voice\Speaker-275-3.wav', res_type='kaiser_fast')
    
    #Displaying the Audio Wave
    plt.figure(figsize=(12, 4))
    librosa.display.waveplot(X2, sr=sr2)
    
    #Passing both the above input to the Feature Extraction function
    combft2 = resize_mfcc(X2 , sr2)
    print(combft2.shape)
    
    #Assigning the Target array and reshaping it to the size of feature array
    val_y=np.asarray(0)
    val_y2=np.resize(val_y,(2239,))
    
    ##Modelling the Machine
    
    #Building the model
    model = Sequential()
    model.add(Dense(250, input_shape=(1,),activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(455, activation='relu'))
    #model.add(Dropout(0.18))
    #model.add(Dense(260, activation='relu'))
    model.add(Dense(470, activation='relu'))
    #model.add(Dropout(0.1))
    model.add(Dense(475, activation='relu'))
    model.add(Dense(480, activation='relu'))
    #Flattening the input
    model.add(layers.Flatten())
    #model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    
    #Training our model
    model.fit(combft, z2, epochs=15)
    
    #Evaluating our model
    val_loss,val_acc = model.evaluate(combft2,val_y2)
    print ("\n validation loss is :", val_loss)
    print ("\n validation accuracy is :", val_acc*100,"%")
    
    if val_acc>=0.85:
        print("\n\n\n\n\n\nSpeaker Identified\n\n\n\n")
    else:
        print("\n\n\n\n\n\nSpeaker Not Identified\n\n\n\n\n")
        
    
