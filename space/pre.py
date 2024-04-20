# Preprocessing Pipeline
 ## Build 2D spectrograms
import random
import torchaudio
from torchaudio import transforms
import torch

class AudioHandler():
    @staticmethod
    def open(file_path):
        # throw away sample rate since it is a constant
        signal, _ = torchaudio.load(file_path)
        return signal
    
    @staticmethod
    def pad(signal, audio_length):
        new_signal = torch.zeros((1, audio_length))
        new_signal[0, 48000 - len(signal):] = signal
        
        return new_signal
    
    @staticmethod
    def time_shift(signal, shift_limit):
        """ Data augmentation: shift forward or back by a bit. """
        signal_length = signal.shape[1]
        shift = int(random.random() * shift_limit * signal_length)
        return signal.roll(shift)
    
    @staticmethod
    def mfcc(signal, n_mfcc=13, n_mels=40, n_fft=2048, hop_len=None):
        mfcc_transform = transforms.MFCC(
            sample_rate=48000,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'n_mels': n_mels,
                'hop_length': hop_len
            }
        )
        mfccs = mfcc_transform(signal)
        return mfccs