from __future__ import division
import os
print(os.getcwd())
import csv
import json
import librosa
import numpy as np
import torchaudio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import soundfile
from soundfile import SoundFile
import time
#print(torch.__version__, torch.backends.cudnn.version(), torch.backends.cudnn.enabled)
import IPython.display as ipd


from Room_simulation2 import Audio_Set, OnlineSimulationDataset, RandomTruncate, fs, V


# -------------- parameter ---------------
LOCATA=False
TEST=False

USE_24K=True

fs=48000



def generate_one_hot_label(lenth, label):
    vector = np.zeros(lenth)
    vector[label] = 1
    return vector

# --------------read the classes hierarchy from json file ------------------- 
include_voice_class = [
"air_conditioner", "car_horn",  "children_playing", 
"dog_bark", "drilling", "engine_idling", "gun_shot", 
"jackhammer", "siren", "street_music"
]

voice_class = [ "car_horn", "dog_bark"]

noise_class = ["air_conditioner",  "children_playing",
"engine_idling", "gun_shot", 
"jackhammer", "siren", "street_music", "drilling"]

audio_folder = "../UrbanSound/data/"

num_classes = len(include_voice_class)
numbers_voice = np.zeros(len(voice_class))
numbers_noise = np.zeros(len(noise_class))
class_index = 0

voice_data = []
noise_data = []
MAX_VOICE_NUM = 160

for class_index in range(0, len(voice_class)):
    c = voice_class[class_index]
    path = audio_folder +  c 
    for x in os.listdir(path):

        if x.endswith(".csv"):
            #print(x)
            with open(path + '/' + x) as f:
                reader = csv.reader(f)  
                filename = c + '/' + x[:-4]
                #print(path + '/' + filename + ".wav")
                if os.path.exists(audio_folder + '/' + filename + ".wav"):
                    filename +=  ".wav"
                elif os.path.exists(audio_folder + '/' + filename + ".aif"):
                    filename += ".aif"
                elif os.path.exists(audio_folder + '/' + filename + ".aiff"):
                    filename += ".aiff"
                elif os.path.exists(audio_folder + '/' + filename + ".wav"):
                    filename += ".wav"
                elif os.path.exists(audio_folder + '/' + filename + ".flac"):
                    filename += ".flac"      
                else: continue              
                    
                for content in reader:
                    #content = next(reader)
                    begin, end, quality, class_name = content
                    if quality == '2': continue
                    #print(begin,type(begin), end, quality)
                    #print(class_name)
                    if class_name in voice_class:
                        label_index = voice_class.index(class_name)
                    else: continue
                    if numbers_voice[label_index] >= MAX_VOICE_NUM: continue
                    label = generate_one_hot_label(len(voice_class), label_index)
                    data = [filename, label, float(begin), float(end)]
                    voice_data.append(data)
                    numbers_voice[label_index] += 1

for class_index in range(0, len(noise_class)):
    c = noise_class[class_index]
    path = audio_folder +  c 
    for x in os.listdir(path):

        if x.endswith(".csv"):
            #print(x)
            with open(path + '/' + x) as f:
                reader = csv.reader(f)     
                #content = next(reader)
                filename = c + '/' + x[:-4]
                
                if os.path.exists(audio_folder + '/' + filename + ".wav"):
                    filename +=  ".wav"
                else: continue

                for content in reader:
                    #content = next(reader)
                    begin, end, quality, class_name = content
                    #if quality == '2': continue
                    #print(begin,type(begin), end, quality)
                    #print(class_name)
                    if class_name in noise_class:
                        label_index = noise_class.index(class_name)
                    else: continue
                    #label = generate_one_hot_label(len(voice_class), label_index)
                    label = np.zeros((1, len(voice_class)))
                    data = [filename, label, float(begin), float(end)]
                    noise_data.append(data)
                    numbers_noise[label_index] += 1

print(numbers_voice, numbers_noise)
print("Voice WAV FILe number: ", len(voice_data))
print("Noise WAV FILe number: ", len(noise_data))

simulation_config={
    'seed':3,
    'min_snr':5,
    'max_snr':25,
    'special_noise_ratio':0.9,
    'pure_ratio': 0.8,
    'source_dist':[0.15, 0.4, 0.4, 0.05],
    'min_angle_diff':25,
    'max_rt60': 0.4,
    'max_room_dim':[7,7,4],
    'min_room_dim':[3,3,2],
    'min_dist': 0.8,
    'min_gap': 1.2,
    'max_order':8,
    'randomize_material_ratio':0.5,
    'max_latency':0.5,
    'random_volume_range': [0.7, 1],
    'low_volume_ratio': 0.04,
    'low_volume_range': [0.01, 0.02],
    'angle_dev': 0.0, # ~5 degree
    'no_reverb_ratio':0.005
}



def generate_data(if_TEST, seg_length = 3, TRAIN_NUM = 5400, TEST_NUM= 600):
    truncator=RandomTruncate(143328, 5, None)#(seg_length*fs, 5, 0.4)

    voice_set = Audio_Set(audio_folder, voice_data)
    noise_set = Audio_Set(audio_folder, noise_data)

    #ipd.Audio(voice_set[0][0], rate=fs)
    #ipd.Audio(noise_set[0][0], rate=fs)
    # no cache for the original dataset

    train_dataset=OnlineSimulationDataset(voice_set, noise_set, TRAIN_NUM, simulation_config, truncator, "./bf_cache/train/")
    test_dataset=OnlineSimulationDataset(voice_set, noise_set, TEST_NUM, simulation_config, truncator, "./bf_cache/test/")


    if if_TEST:
        total, vector, gt=train_dataset[11]
        data1 = total[0,:]
        data1 = data1[:, np.newaxis]
        data1 = np.repeat(data1, 2, axis = 1)
        with SoundFile('total.wav', 'w', fs, 2, 'PCM_24') as f:
            f.write(data1)

        data1 = gt[0,:]
        data1 = data1[:, np.newaxis]
        data1 = np.repeat(data1, 2, axis = 1)
        with SoundFile('gt.wav', 'w', fs, 2, 'PCM_24') as f:
            f.write(data1)
        #soundfile.write("total.wav", total, fs, format = 'WAV')
        #soundfile.write("gt.wav", gt, fs, format = 'WAV')
        #plt.figure()
        #plt.plot(total[0])
        #plt.figure()
        #plt.plot(gt[0])
        print(vector)
        #plt.show()
    
    return train_dataset, test_dataset

if __name__ == "__main__":
    soundfile.check_format('WAV', 'PCM_24')
    generate_data(1, seg_length = 3, TRAIN_NUM = 20, TEST_NUM = 4)
