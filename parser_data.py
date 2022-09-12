import json
import os
import sys 
from tkinter.ttk import LabeledScale
import librosa
from matplotlib.pyplot import axis
import numpy as np
import time
from scipy.signal import find_peaks
from scipy.fft import fft

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
WORKSPACE = os.path.dirname(FILE_PATH)

sys.path.insert(0, os.path.join(WORKSPACE, "instrumentsDataset"))


DATASET_PATH = "C:/Users/Juanma/Desktop/ZapadAPP/Workspace/instrumentsDatasets/NoteTrain"
JSON_PATH = "data_note.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 3 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
NFFT = 512
SHAPE = 100
HOP_LENGTH = 1024
N_MELS_BAND = 50

def isWavDir(datapath):
    list_dir = os.listdir(datapath)
    for x in list_dir:
        if '.wav' in x :
            return True
    return False

def generateLabel(datapath):
     dir_labels = datapath.split("\\")
     size_dir = len(dir_labels)

    # semantic_label = dir_labels[size_dir - 2] + "-"+ dir_labels[size_dir - 1]
     semantic_label = dir_labels[size_dir -2 ] + "-" + dir_labels[size_dir - 1]
    
     return semantic_label

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


def save_information_notes(dataset_path, json_path):
    
    data = {
        "mapping": [],
        "labels": [],
        "mel_spec": []
    }
    count_label = 0 
    total_files = 0
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        # ensure we're processing a chord sub-folder level
        if dirpath is not dataset_path and isWavDir(dirpath):
            #genero el label.
            label = generateLabel(dirpath)
            data["mapping"].append(label)
		    
            print("\nProcessing: {}\nLabel Assign: {}\n".format(label, count_label))
		
            # process all audio files in chord sub-dir
            for f in filenames:

		        # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
                duration = librosa.get_duration(y=signal, sr= sample_rate)
                record = os.path.split(file_path)[1]
                #onSet Detect Here
                if duration > 3.0:
                    onset_frames = librosa.onset.onset_detect(y=signal, sr=sample_rate, wait=1, pre_avg=1, post_avg=1, pre_max=1, post_max=1)

                    samples = librosa.frames_to_samples(onset_frames)
    			    # filter lower samples
                    
                    filteredSamples = filterLowSamples(samples)
   			        # get indexes of all samples in the numpy array
                    indexes = np.where(filteredSamples>0)
                    length = len(indexes[0])
                    
                    j = 0

                    for i in indexes[0]:
                        j = i
                        if j < length-1:
                            onset_signal = signal[filteredSamples[j]:filteredSamples[j+1]]
                        elif j == length-1:
                            onset_signal = signal[filteredSamples[j]:]
                        
                        mel_spec = librosa.feature.melspectrogram(y=onset_signal, sr=sample_rate, n_mels=N_MELS_BAND,fmin=100,fmax= 650)
                        
                        if not correctShape(mel_spec.shape[1]) :
                            mel_spec =  normalizeShape(mel_spec)
                            # generate x values (frequencies)
                        if correctShape(mel_spec.shape[1])  :
                            data["mel_spec"].append(mel_spec.tolist())
                            data["labels"].append(count_label)
                            print("{}, file:{} mel_shape:{}".format(label, record,mel_spec.shape))
                            total_files = total_files + 1
                else:
                        mel_spec = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_mels=N_MELS_BAND,fmin=100 ,fmax= 650)

                        if not correctShape(mel_spec.shape[1]) :
                            mel_spec =  normalizeShape(mel_spec)
                            # generate x values (frequencies)
                        data["mel_spec"].append(mel_spec.tolist())
                        data["labels"].append(count_label)
                        print("{}, file:{} mel_shape:{}".format(label, record,mel_spec.shape))
                        total_files = total_files + 1
            count_label = count_label + 1 

    # save CHROMA to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
    return total_files
    
def listFrecuencies(frec,f_bins,X_mag,peaks):
    frec_list = []
    i = 0 
    while i < 3000 :
        y_peak = peaks[i]
        peak_i = np.where(X_mag[:f_bins] == y_peak)
        frec_list.append(float(frec[peak_i[0]]))
        i = i + 1
        
     
    return frec_list
## Function that filters lower samples generated by input noise.
def filterLowSamples(samples):
    # find indexes of all elements lower than 2000 from samples
    indexes = np.where(samples < 2000)
    # remove elements for given indexes
    return np.delete(samples, indexes)


if __name__ == "__main__":
    init_time = time.time()
    total_files = save_information_notes(DATASET_PATH, JSON_PATH)
    finish_time = time.time()
    total_time = finish_time - init_time
    seconds = int(total_time %60)
    minutes = int((total_time/60)%60)
    hours = int(total_time /3600)
    print("RESUME\n")
    print("Total files processing: {}\n".format(total_files))
    print("Time processing: {}:{}:{}".format(hours,minutes,seconds))
