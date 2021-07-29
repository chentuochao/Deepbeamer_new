from __future__ import division
import os
print(os.getcwd())
import csv
import json

import numpy as np
import torchaudio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import time
#print(torch.__version__, torch.backends.cudnn.version(), torch.backends.cudnn.enabled)
import IPython.display as ipd


from Room_simulation import Audio_Set, OnlineSimulationDataset, RandomTruncate, fs, V

# -------------- parameter ---------------
LOCATA=False
TEST=False

USE_24K=True

# --------------read the classes hierarchy from json file ------------------- 
include_voice_class = ["Singing", "Speech",  "Motor vehicle (road)", "Rail transport", "Tools",
 "Plucked string instrument", "Violin, fiddle","Keyboard (musical)", "Drum", "Dog", "Bird", "Water", "Wind" ]
 
num_classes = len(include_voice_class)
#include_voice_class = [ "Plucked string instrument"]
valid_classes = {}

def find_all_children(classes_lists, index, all_ids):
    #print(index)
    item = classes_lists[index]
    id = item["id"]
    all_ids.append(id)
    children = item["child_ids"]

    #print(index, id, children)
    if len(children) == 0: return index + 1
    else:
        new_index = index + 1
        for i in range(0, len(children)):
            new_index = find_all_children(classes_lists, new_index, all_ids)
        return new_index

json_file = "ontology.json"

with open(json_file) as f:
    classes_lists = json.load(f)
    index = 0

    for item in classes_lists:
        name = item["name"]
        if name in include_voice_class:
            print(name)
            all_ids = []
            new_index = find_all_children(classes_lists, index, all_ids)
            valid_classes[name] = all_ids    
        index += 1 
    #print(valid_classes)
    f.close()    


# --------------read the audioset from audio files -------------------

with open("eval_segments.csv") as f:
    reader = csv.reader(f)
    header1 = next(reader)
    header2 = next(reader)
    #print(header1)
    #print(header2)

    numbers = np.zeros(num_classes)
    mono_numbers = np.zeros(num_classes)
    pure_numbers = np.zeros(num_classes)

    voice_data = []
    noise_data = [] 
    pure_data = []
    mix_data = []

    for row in reader:
        #if(line_count > 5): break
        temp_labels = np.zeros(num_classes)
        filename = row[0] + '_' + row[1][1:] + "_out.wav"
        #print("audioset/" + filename)
        if not os.path.exists("audioset/" + filename): continue
        #print(filename)
        labels = row[3:]
        valid_labels = []
        with_noise = False

        for l in labels:
            #print(l, l[:-1], l[2:] )
            if_find = False
            for i in range(0, num_classes):
                name = include_voice_class[i]
                if l in valid_classes[name] or l[:-1] in valid_classes[name] or l[2:] in valid_classes[name] :
                    if_find = True
                    numbers[i] += 1
                    temp_labels[i] = 1
                    if i not in valid_labels: valid_labels.append(i)
            if not if_find: with_noise = True

        if len(valid_labels) <= 0: noise_data.append((filename, temp_labels))
        else:
            if len(valid_labels) == 1:  
                mono_numbers[valid_labels[0]] += 1
                pure_data.append((filename, temp_labels))
                if with_noise == False: 
                    pure_numbers[valid_labels[0]] += 1
                    #pure_data.append((filename, temp_labels))
            else: mix_data.append((filename, temp_labels))
            voice_data.append((filename, temp_labels))

    for i in range(0, num_classes):
        print(include_voice_class[i], numbers[i], mono_numbers[i], pure_numbers[i])
    f.close()
    print("Voice WAV FILe number: ", len(voice_data))
    print("Noise WAV FILe number: ", len(noise_data))
# ------------ generate the simulated voice -----------
# test simulation dataset and truncate
simulation_config={
    'seed':3,
    'min_snr':5,
    'max_snr':25,
    'special_noise_ratio':0.9,
    'pure_ratio': 0.5,
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
    truncator=RandomTruncate(seg_length*fs, 5, 0.4)

    voice_set = Audio_Set('audioset', voice_data)
    noise_set = Audio_Set('audioset', noise_data)
    pure_set  = Audio_Set('audioset', pure_data)
    mix_set = Audio_Set('audioset', mix_data)

    ipd.Audio(voice_set[0][0], rate=fs)
    ipd.Audio(noise_set[0][0], rate=fs)
    # no cache for the original dataset

    train_dataset=OnlineSimulationDataset(pure_set, mix_set, noise_set, TRAIN_NUM, simulation_config, truncator, "./bf_cache/train/")
    test_dataset=OnlineSimulationDataset(pure_set, mix_set, noise_set, TEST_NUM, simulation_config, truncator, "./bf_cache/test/")


    if if_TEST:
        total, vector, gt=train_dataset[12]
        plt.figure()
        plt.plot(total[0])
        plt.figure()
        plt.plot(gt[0])
        print(vector)
        plt.show()
    
    return train_dataset, test_dataset

