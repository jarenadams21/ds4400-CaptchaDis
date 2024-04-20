from torch.utils.data import Dataset, DataLoader
from pre import AudioHandler
import random

file_paths = './data/'

class AudioMNIST(Dataset):
    def __init__(self, load_entire_filetree=False, subset_size=10000):
        self.files = self._build_files(subset_size)
        self.audio_len = 48000
        self.shift_ptc = 0.4
        
    def _build_files(self, subset_size):
        files = {}
        index = 0
        all_files = []
        for ii in range(1, 61):
            num = "0%d" % ii if ii < 10 else "%d" % ii
            for jj in range(50):
                for kk in range(10):
                    all_files.append([
                        file_paths + num + "/%d_%s_%d.wav" % (kk, num, jj),
                        kk
                    ])
        
        selected_files = random.sample(all_files, subset_size)
        for file in selected_files:
            files[index] = file
            index += 1
        return files
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        signal = AudioHandler.open(self.files[idx][0])[0]
        pad = AudioHandler.pad(signal, self.audio_len)
        shift = AudioHandler.time_shift(pad, self.shift_ptc)
        mfcc = AudioHandler.mfcc(shift)
        
        return mfcc, self.files[idx][1]
