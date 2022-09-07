from array import ArrayType
from itertools import count
import keras
import librosa
import librosa.display 
import numpy as np
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys 
from scipy.signal import find_peaks
from scipy.fft import fft
import time 
import pyaudio
import webrtcvad
import pyaudio
import wave
import os
from warnings import simplefilter
import time



FILE_PATH = os.path.dirname(os.path.realpath(__file__))
WORKSPACE = os.path.dirname(FILE_PATH)
MY_MODEL = keras.models.load_model('modelo-notas-v01.h5')
sys.path.insert(0, os.path.join(WORKSPACE, "input_parser"))



FILE_PATH = "Predict\Guitar_C5_1662080816.4392762.wav"
ROOT_PATH = "Predict"
DATASET_PATH = "Data"
JSON_PATH = "data_note.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 3 # measured in seconds
NFFT = 512
hop_length = 1024
SHAPE = 100
N_MELS_BANDS = 50

CATEGORIES = ["A2","A3","A4","B2","B3","B4","C3","C4","C5","D3","D4","D5","E2","E3","E4",
              "F2","F3","F4","G2","G3","G4",
              "A3","A4","A5","B3","B4","B5","C3","C4","C5","D3","D4","D5","E3","E4","E5",
              "F3","F4","F5","G3","G4","G5",]

INSTRUMENT = ["Guitar","Piano"]

def correctShape(mel_shape):
    return mel_shape == SHAPE

def normalizeShape(mel_mat):
    nums = 0
    #init_shape tiene la dimension de columnas. 
    init_shape= mel_mat.shape[1]
    #Me fijo cuantas columnas faltan por rellenar
    nums = SHAPE - init_shape
    #itero nums copiando el anterior
    arreglo = np.array(mel_mat[:,init_shape-1])
    i = 0
    if nums > 0 :
        while i < nums :
            mel_mat= np.column_stack((mel_mat,arreglo))  
            i = i +1 
    else:
        #print("MY SHAPE IS: {}".format(mel_mat.shape[1]))
        mel_mat = np.array(mel_mat[:,: SHAPE])
        #print("NOW MY SHAPE IS: {}".format(mel_mat.shape[1]))

             
    return mel_mat

def checkBach(testing_path, note, instrument):
    test_instrument = "init"
    test_note = "init"
    count = 0
    success = 0
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(testing_path)):
     
        for f in filenames:
		# load audio file
            file_path = os.path.join(dirpath, f)
            signal, sr = librosa.load(file_path)
            test_instrument, test_note = getNoteandInstrumentFromRNN(signal,sr)
            if test_instrument == instrument and test_note == chord:
                success = success +1
            count = count + 1 
    print("****RESUME****\n")
    print("Note Selected: {}\n".format(note))
    print("Instrument Selected: {}\n".format(instrument))
    print("Total in Bach: {}\n".format(count))
    print("Total test success: {}\n".format(success))
    print("%\Accuracy: {}%\n".format(success*100/count))
    return

def getNoteandInstrumentFromRNN(signal, sample_rate):
    
    instrument = INSTRUMENT[1]
    note = "not recognize "
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_mels=N_MELS_BANDS,fmax=1000) 
    if not correctShape(mel_spec.shape[1]):
        mel_spec =  normalizeShape(mel_spec)

    if correctShape(mel_spec.shape[1]):
        mel_reshape = tf.reshape(mel_spec, [ 1,N_MELS_BANDS,SHAPE ])
        my_prediction = MY_MODEL.predict(mel_reshape)
        index = np.argmax(my_prediction)
        note = CATEGORIES[index]
    
        if index < 21 :
            instrument = INSTRUMENT[0]
        print("Instrument {}\n".format( instrument))
        print("Note {}\n".format( note))
    
    return instrument, note


def convertToNote(val) :
    nota = str.replace(str.replace(val, "['", ""), "']", "")
    if len(nota) == 3 :
        nota = nota[0] + "#" + nota[2]

    return nota

def largeAudioWithOnset(FILE_PATH,note,instrument):
      
    y, sr = librosa.load(FILE_PATH)
    ok_test = 0
    audio_onsets = 0
    test_instrument = ""
    test_note = ""
    ok_test_note = 0
    ok_test_instrument = 0
    onset_frames = librosa.onset.onset_detect(y, sr=sr, wait=1, pre_avg=1, post_avg=1, pre_max=1, post_max=1)
    samples = librosa.frames_to_samples(onset_frames)

    # filter lower samples
    filteredSamples = filterLowSamples(samples)

    # get indexes of all samples in the numpy array
    indexes = np.where(filteredSamples>0)

    length = len(indexes[0])
    print("len samples {}".format(length))
    j = 0
 
    for i in indexes[0]:
        j = i
        if j < length-1:
            test_instrument, test_note = getNoteandInstrumentFromRNN(y[filteredSamples[j]:filteredSamples[j+1]],sr)    
        elif j == length-1:
            test_instrument, test_note = getNoteandInstrumentFromRNN(y[filteredSamples[j]:],sr)
        
        if test_instrument == instrument : 
            ok_test_instrument = ok_test_instrument + 1
        if  test_note == note:
            ok_test_note = ok_test_note + 1
        audio_onsets = audio_onsets + 1
    ok_test = ok_test_note
    accuracy = ((ok_test_note  + ok_test_instrument)/2)/audio_onsets
    
    print("****RESUME****\n")
    print("Note Selected: {}\n".format(note))
    print("Instrument Selected: {}\n".format(instrument))
    print("Total Onsets: {}\n".format(audio_onsets))
    print("Total test notes success: {}\n".format(ok_test_note))
    print("Total test instrument success: {}\n".format(ok_test_instrument))
    print("%\Accuracy: {}%\n".format(ok_test*100/audio_onsets))

## Function that filters lower samples generated by input noise.
def filterLowSamples(samples):
    # find indexes of all elements lower than 2000 from samples
    indexes = np.where(samples < 2000)
    # remove elements for given indexes
    return np.delete(samples, indexes)


def getOctaveFromPeak(peak):
                
    nota = str(librosa.hz_to_note(peak))
    nota = convertToNote(nota)
    octava = nota[1]
    if octava == "#":
        octava = nota[2]
    return int(octava)
   

if __name__ == "__main__":
    
    print("***WELCOME***")
    noteIndex = input("\nSelect a note:\n[0] - C\n[1] - D\n[2] - E\n[3] - F\n[4] - G\n[5] - A\n[6] - B\n>")

    note = ""
    match noteIndex:
        case "0":
            note = "C"
        case "1": 
            note = "D"
        case "2":
            note = "E"
        case "3":
            note = "F"
        case "4":
            note = "G"
        case "5":
            note = "A"
        case "6":
            note = "B"            
        case _:
            print("Please don't be retarded")
            quit()
    scale = input("\nSelect a scale from 2 to 6: ")
    instrumentIndex = input ("\nSelect instrument\n[0] - Guitar\n[1] - Piano\n->:")
    instrument = INSTRUMENT[int(instrumentIndex)]
    testIndex = input("\nSelect a test:\n[0] - Batch\n[1] - Only One\n[2] - Onset\n[3] - Exit:\n>:")
    note = note + str(scale)
    match testIndex:
        case "0":
            checkBach(FILE_PATH,note,instrument)    
        case "1": 
            signal, sr = librosa.load(FILE_PATH)
            instrument, chord = getNoteandInstrumentFromRNN(signal,sr)
            print("TESTING PREDICTION COME HERE\n")
            print("The instrument is {}\n".format(instrument))
            print("The note is {}\n".format(note))
            print("PLEASE FOLLOW WITH THE TEST! \n")
        case "2":
            largeAudioWithOnset(FILE_PATH,note,instrument)
        case "3":
            print("See you\n")
            quit()

