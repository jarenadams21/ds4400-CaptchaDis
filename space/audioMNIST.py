# DataSet and DataLoader
    ## Create indexed dictionary of filenames and load one batch into memory at a time.
    ## Use multiple workers to speed up training
from torch.utils.data import Dataset
from pre import AudioHandler


file_paths = './data/'


class AudioMNIST(Dataset):
    def __init__(self, load_entire_filetree=False):
        self.files = self._build_files()
        self.audio_len = 48000
        self.shift_ptc = 0.4
        
    def _build_files(self):
        files = {}
        index = 0
        for ii in range(1, 61):
            num = "0%d" % ii if ii < 10 else "%d" % ii
            for jj in range(50):
                for kk in range(10):
                    files[index] = [
                        file_paths + num + "/%d_%s_%d.wav" % (
                            kk, num, jj),
                        kk
                    ]
                    index += 1
                    
        return files
    
    def __len__(self):
        return 30000
    
    def __getitem__(self, idx):
        signal = AudioHandler.open(self.files[idx][0])[0]
        pad = AudioHandler.pad(signal, self.audio_len)
        shift = AudioHandler.time_shift(pad, self.shift_ptc)
        mfcc = AudioHandler.mfcc(shift)
        
        return mfcc, self.files[idx][1]