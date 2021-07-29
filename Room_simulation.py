import numpy as np
import pyroomacoustics as pra

N_MIC=2 # number
R_MIC=0.06

max_rt60=0.3
max_room_dim=[10,10,4]
min_room_dim=[4,4,2]
min_dist=0.8 # dist between mic and person
min_gap=1.2 # gap between mic and walls

fs=48000
V=343

def generate_mic_array(mic_radius: float, n_mics: int, pos):
    """
    Generate a list of Microphone objects
    Radius = 50th percentile of men Bitragion breadth
    (https://en.wikipedia.org/wiki/Human_head)
    """
    R = pra.circular_2D_array(center=[pos[0], pos[1]], M=n_mics, phi0=0, radius=mic_radius)
    R=np.concatenate((R, np.ones((1, n_mics))*pos[2]), axis=0)
    return R

# global mic array:
R_global=generate_mic_array(R_MIC, N_MIC, (0,0,0))


# simulate the room

def get_dist(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def get_angle(px, py):
    return np.arctan2(py, px)

def random2D(range_x, range_y, except_x, except_y, except_r, retry=100):   #???? why we need recursion tp generate 2D pos
    if retry==0: return None
    if except_r<except_x<range_x-except_r and except_r<except_y<range_y-except_r:
        loc=(np.random.uniform(0, range_x), np.random.uniform(0, range_y))
        if get_dist(loc, (except_x, except_y))<except_r:
            return random2D(range_x, range_y, except_x, except_y, except_r, retry-1)
        else:
            return loc
    else:
        return None
        
def simulateRoom(N_source, min_room_dim, max_room_dim, min_gap, min_dist, min_angle_diff=0, retry=20):
    if retry==0: return None
    # return simulated room
    room_dim=[np.random.uniform(low, high) for low, high in zip(min_room_dim, max_room_dim)] # get the random size of room
    R_loc=[np.random.uniform(min_gap, x-min_gap) for x in room_dim]
    source_locations=[random2D(room_dim[0], room_dim[1], R_loc[0], R_loc[1], min_dist) for i in range(N_source)]
    if None in source_locations: return None
    
    angles=[get_angle(p[0]-R_loc[0], p[1]-R_loc[1]) for p in source_locations]
    if N_source>1:
        min_angle_diff_rad=min_angle_diff*np.pi/180
        angles_sorted=np.sort(angles)
        if np.min(angles_sorted[1:]-angles_sorted[:-1])<min_angle_diff_rad or angles_sorted[0]-angles_sorted[-1]+2*np.pi<min_angle_diff_rad:
            return simulateRoom(N_source, min_room_dim, max_room_dim, min_gap, min_dist, min_angle_diff, retry-1)
    
    source_locations=[(x,y,R_loc[2]) for x,y in source_locations]
    
    return (room_dim, R_loc, source_locations, angles) 
    # room_dim --- the size of room, R_loc the position of microphone array center, source_locations -- the position of sources, angles - the angle between source and microphone array center

# Define materials
wall_materials = [
    'hard_surface',
    'brickwork',
    'rough_concrete',
    'unpainted_concrete',
    'rough_lime_wash',
    'smooth_brickwork_flush_pointing',
    'smooth_brickwork_10mm_pointing',
    'brick_wall_rough',
    'ceramic_tiles',
    'limestone_wall'
]
floor_materials = [
    'linoleum_on_concrete',
    'carpet_cotton',
    'carpet_tufted_9.5mm',
    'carpet_thin',
    'carpet_6mm_closed_cell_foam',
    'carpet_6mm_open_cell_foam',
    'carpet_tufted_9m',
    'felt_5mm',
    'carpet_soft_10mm',
    'carpet_hairy',
]

def simulateSound(room_dim, R_loc, source_locations, source_audios, rt60, materials=None, max_order=None):
    # source_audios: array of numpy array
    # L: max of all audios. Zero padding at the end
    # return (all_channel_data (C, L), groundtruth_with_reverb (N, C, L), groundtruth_data (N, C, L), angles (N)
    
    if materials is not None:
        (ceiling, east, west, north, south, floor)=materials
        room = pra.ShoeBox(
            room_dim,
            fs=fs,
            materials=pra.make_materials(
                ceiling=ceiling,
                floor=floor,
                east=east,
                west=west,
                north=north,
                south=south,
            ), max_order=max_order
        )
    else:
        try:
            e_absorption, max_order_rt60 = pra.inverse_sabine(rt60, room_dim)    
        except ValueError:
            e_absorption, max_order_rt60 = pra.inverse_sabine(1, room_dim)
        room=pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=min(max_order_rt60, max_order))
        #room=pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=0)
    
    R=generate_mic_array(R_MIC, N_MIC, R_loc)
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))
    
    length=max([len(source_audios[i]) for i in range(len(source_audios))])
    for i in range(len(source_audios)):
        source_audios[i]=np.pad(source_audios[i], (0, length-len(source_audios[i])), 'constant')
        
    
    for i in range(len(source_locations)):
        room.add_source(source_locations[i], signal=source_audios[i], delay=0)
    
    room.image_source_model()
    premix_w_reverb=room.simulate(return_premix=True) 
    mixed_reverb=room.mic_array.signals
    
    # groundtruth
    room_gt=pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(1.0), max_order=0)
    
    new_angles=np.zeros((len(source_locations),))
    
    R_gt=R[:, 0].reshape((3,1)) #generate_mic_array(0,1,R_loc)
    room_gt.add_microphone_array(pra.MicrophoneArray(R_gt, room.fs))
    
    for i in range(len(source_locations)):
        room_gt.add_source(source_locations[i], signal=source_audios[i], delay=0)
        new_angles[i]=get_angle(source_locations[i][0]-R_gt[0,0], source_locations[i][1]-R_gt[1,0])
    room_gt.compute_rir()
    
    room_gt.image_source_model()
    premix=room_gt.simulate(return_premix=True)
    mixed=room_gt.mic_array.signals
    
    return (mixed, premix_w_reverb, premix, new_angles)

def simulateBackground(background_audio):
    # diffused noise. simulate in a large room
    bg_radius = np.random.uniform(low=10.0, high=20.0)
    bg_theta = np.random.uniform(low=0, high=2 * np.pi)
    H=10
    bg_loc = [bg_radius * np.cos(bg_theta), bg_radius * np.sin(bg_theta), H]

    # Bg should be further away to be diffuse
    left_wall = np.random.uniform(low=-40, high=-20)
    right_wall = np.random.uniform(low=20, high=40)
    top_wall = np.random.uniform(low=20, high=40)
    bottom_wall = np.random.uniform(low=-40, high=-20)
    height = np.random.uniform(low=20, high=40)
    corners = np.array([[left_wall, bottom_wall], [left_wall, top_wall],
                    [   right_wall, top_wall], [right_wall, bottom_wall]]).T
    absorption = np.random.uniform(low=0.5, high=0.99)
    room = pra.Room.from_corners(corners,
                                 fs=fs,
                                 max_order=10,
                                 materials=pra.Material(absorption))
    room.extrude(height)
    mic_array = generate_mic_array(R_MIC, N_MIC, (0,0,H))
    room.add_microphone_array(pra.MicrophoneArray(mic_array, fs))
    room.add_source(bg_loc, signal=background_audio)

    room.image_source_model()
    room.simulate()
    return room.mic_array.signals


from util import power, mix, find_all_label_combination
from torch.utils.data import Dataset
import numpy as np
import pyroomacoustics as pra
import os
import librosa

class Audio_Set:
    def __init__(self, path, data_list, shuffle=True, test=True, train=True):
        self.audio_files = data_list
        self.path = path
        if shuffle:
            np.random.shuffle(self.audio_files)
    def __len__(self):
        return len(self.audio_files)
    def __read_audio(self,file):
        audio, _ = librosa.load(self.path + '/' + file, sr=fs, mono=True)
        return audio
    def __getitem__(self, idx):
        return self.__read_audio(self.audio_files[idx][0]),  np.array(self.audio_files[idx][1])
    def get_batch(self, idx1, idx2):
        mic_sig_batch = []
        for idx in range(idx1, idx2):
            mic_sig_batch.append(self.__read_audio(self.audio_files[idx][0]),  np.array(self.audio_files[idx][1]))
            
        return mic_sig_batch




class OnlineSimulationDataset(Dataset):
    def __init__(self, voice_collection_pure, voice_collection_mix, noise_collection, length, simulation_config, truncator, cache_folder, cache_max=None):
        self.voices_pure=voice_collection_pure
        self.voices_mix=voice_collection_mix
        self.noises=noise_collection
        
        self.length=length
        self.seed=simulation_config['seed']
        self.additive_noise_min_snr=simulation_config['min_snr']
        self.additive_noise_max_snr=simulation_config['max_snr']
        self.special_noise_ratio=simulation_config['special_noise_ratio']
        self.pure_ratio = simulation_config['pure_ratio']
        self.source_dist=simulation_config['source_dist']
        self.min_angle_diff=simulation_config['min_angle_diff']
        self.max_rt60=simulation_config['max_rt60'] # 0.3s
        self.min_rt60=0.15 # minimum to satisfy room odometry
        self.max_room_dim=simulation_config['max_room_dim'] # [10,10,4]
        self.min_room_dim=simulation_config['min_room_dim'] #[4,4,2]
        self.min_dist=simulation_config['min_dist'] # 0.8, dist between mic and person
        self.min_gap=simulation_config['min_gap'] # 1.2, gap between mic and walls
        self.max_order=simulation_config['max_order']
        self.randomize_material_ratio=simulation_config['randomize_material_ratio']
        self.max_latency=simulation_config['max_latency']
        self.random_volume_range=simulation_config['random_volume_range'] # max and min volume ratio for sources
        
        self.low_volume_ratio=simulation_config['low_volume_ratio']
        self.low_volume_range=simulation_config['low_volume_range']
        self.angle_dev=simulation_config['angle_dev']
        
        self.no_reverb_ratio=simulation_config['no_reverb_ratio']
        
        self.truncator=truncator
        self.cache_folder=cache_folder
        self.cache_history=[]
        self.cache_max=cache_max
        
    def __seed_for_idx(self,idx):
        return self.seed+idx
    
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        # return format: 
        # (
        # mixed multichannel audio, (C,L)
        # array of groundtruth with reverb for each target, (N, C, L)
        # array of direction of targets, (N,)
        # array of multichannel ideal groundtruths for each target, (N, C, L)
        # noise (C, L)
        # )
        # check cache first
        
        if idx>=self.length:
            return None
        #print("sssssssssss")
        if self.cache_folder is not None:
            cache_path=self.cache_folder+'/'+str(idx)+'-'+str(self.seed)+'.npz'
            
            if cache_path not in self.cache_history:
                self.cache_history.append(cache_path)
            
                if self.cache_max is not None and self.cache_max==len(self.cache_history):
                    # delete first one
                    first=self.cache_history[0]
                    os.remove(first)
                    self.cache_history=self.cache_history[1:]
                
            if os.path.exists(cache_path):
                cache_result=np.load(cache_path, allow_pickle=True)['data']
                return cache_result[0], cache_result[1], cache_result[2]
        else:
            cache_path=None
        #print("sssssssssss1")
        
        np.random.seed(self.__seed_for_idx(idx))
        n_source=np.random.choice(np.arange(len(self.source_dist)), p=self.source_dist)+1
        
        n_pure = 0
        n_mix = 0
        for i in range(0, n_source):
            if np.random.rand()<self.pure_ratio: n_pure += 1
            else: n_mix += 1


        room_result=simulateRoom(n_source, self.min_room_dim, self.max_room_dim, self.min_gap, self.min_dist, self.min_angle_diff)
        if room_result is None:
            return self.__getitem__(idx+1) # backoff
        room_dim, R_loc, source_loc, source_angles=room_result


        # generate the room size and location of source and microphone center
        random_indexes_source = np.random.choice(len(self.voices_pure), n_pure)
        voices_p=[self.truncator.process(self.voices_pure[vi][0]) for vi in random_indexes_source]
        voice_labels_p = [self.voices_pure[vi][1] for vi in random_indexes_source]

        random_indexes_source = np.random.choice(len(self.voices_mix), n_mix)
        voices_m=[self.truncator.process(self.voices_mix[vi][0]) for vi in random_indexes_source]
        voice_labels_m = [self.voices_mix[vi][1] for vi in random_indexes_source]

        voices = voices_p + voices_m
        voice_lable = voice_labels_p + voice_labels_m
        label_options = find_all_label_combination(voice_lable)

        select_vector, mix_indexes = label_options[np.random.choice(len(label_options))]

        # normalize voice
        voices=[v/np.max(np.abs(v)) for v in voices]
        
        voices=[v*np.random.uniform(self.random_volume_range[0], self.random_volume_range[1]) for v in voices]

        #if np.random.rand()<self.low_volume_ratio:
        #    voices[0]*=np.random.uniform(self.low_volume_range[0], self.low_volume_range[1])
        
        if self.special_noise_ratio>np.random.rand():
            noise=self.truncator.process(self.noises[np.random.choice(len(self.noises))][0])
        else:
            noise=np.random.randn(self.truncator.get_length())

        no_reverb=(np.random.rand()<self.no_reverb_ratio)
        max_order=0 if no_reverb else self.max_order
        if self.randomize_material_ratio>np.random.rand():
            ceiling, east, west, north, south = tuple(np.random.choice(wall_materials, 5))  # sample material
            floor = np.random.choice(floor_materials)  # sample material
            mixed, premix_w_reverb, premix, new_angles=simulateSound(room_dim, R_loc, source_loc, voices, 0, (ceiling, east, west, north, south, floor), max_order)
        else:
            rt60=np.random.uniform(self.min_rt60, self.max_rt60)
            mixed, premix_w_reverb, premix, new_angles=simulateSound(room_dim, R_loc, source_loc, voices, rt60, None, max_order)
        #generate the simulated speech sound
        
        gt = np.sum(premix[mix_indexes, :, :], axis=0) # generate groundtruth
        #print(mixed.shape, gt.shape)

        background=simulateBackground(noise)
        #generate the simulated noise sound
        snr=np.random.uniform(self.additive_noise_min_snr, self.additive_noise_max_snr)
        
        # trucate to the same length
        mixed=mixed[:, :self.truncator.get_length()]
        background=background[:, :self.truncator.get_length()]
        
        total, background=mix(mixed, background, snr)
        
        #new_angles[0]+=(np.random.rand()*2-1)*self.angle_dev

        
        # save cache
        if cache_path is not None:
            np.savez_compressed(cache_path, data=[total, select_vector, gt])
        
        return total, select_vector, gt
        #return mixed, premix_w_reverb, source_angles, premix, background

class RandomTruncate:
    def __init__(self, target_length, seed, power_threshold=None):
        self.length=target_length
        self.seed=seed
        self.power_threshold=power_threshold
        np.random.seed(seed)
    
    def process(self, audio):
        # if there is a threshold
        if self.power_threshold is not None:
            # smooth
            power=np.convolve(audio**2, np.ones((32,)), 'same')
            avgpower=np.mean(power)
            for i in range(len(power)):
                # threshold*mean_power
                if power[i]>avgpower*self.power_threshold:
                    #print(i, power[i], avgpower)
                    # leave ~=0.3s of start
                    fs=48000
                    audio=audio[max(0, i-int(0.3*fs)):]
                    break
        if len(audio)<self.length:
            nfront=np.random.randint(self.length-len(audio))
            return np.pad(audio, (nfront, self.length-len(audio)-nfront), 'constant')
        elif len(audio)==self.length:
            return audio
        else:
            start=np.random.randint(len(audio)-self.length)
            return audio[start:start+self.length]
        
    def get_length(self):
        return self.length