import math
import os
import random
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
import torchaudio
from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize
from scipy.io.wavfile import read


def load_wav(full_path):
    #sampling_rate, data = read(full_path)
    #return data, sampling_rate
    data, sampling_rate = librosa.load(full_path, sr=None)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}

class LogMelSpectrogram(torch.nn.Module):
    def __init__(self, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
        super().__init__()
        self.melspctrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            win_length=win_size,
            hop_length=hop_size,
            center=center,
            power=1.0,
            norm="slaney",
            onesided=True,
            n_mels=num_mels,
            mel_scale="slaney",
            f_min=fmin,
            f_max=fmax
        )
        self.n_fft = n_fft
        self.hop_size = hop_size

    def forward(self, wav):
        wav = F.pad(wav, ((self.n_fft - self.hop_size) // 2, (self.n_fft - self.hop_size) // 2), "reflect")
        mel = self.melspctrogram(wav)
        logmel = torch.log(torch.clamp(mel, min=1e-5))
        return logmel


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    # print("Padding by", int((n_fft - hop_size)/2), y.shape)
    # pre-padding
    n_pad = hop_size - ( y.shape[1] % hop_size )
    y = F.pad(y.unsqueeze(1), (0, n_pad), mode='reflect').squeeze(1)
    # print("intermediate:", y.shape)

    y = F.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)
    
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = spec.abs().clamp_(3e-5)
    # print("Post: ", y.shape, spec.shape)

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def get_dataset_filelist(a):
    train_df = pd.read_csv(a.input_training_file)
    valid_df = pd.read_csv(a.input_validation_file)
    return train_df, valid_df


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, audio_root_path=None, feat_root_path=None, use_alt_melcalc=False):
        self.audio_files = training_files
        if shuffle:
            self.audio_files = self.audio_files.sample(frac=1, random_state=1234)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.audio_root_path = Path(audio_root_path)
        self.feat_root_path = Path(feat_root_path)
        self.alt_melspec = LogMelSpectrogram(n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax)
        self.use_alt_melcalc = use_alt_melcalc

    def __getitem__(self, index):
        row = self.audio_files.iloc[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav(row.audio_path)
            if not self.fine_tuning:
                audio = normalize(audio) * 0.95
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.tensor(audio, dtype=torch.float32)
        audio = audio.unsqueeze(0)

        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start+self.segment_size]
                else:
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

            if self.use_alt_melcalc:
                mel = self.alt_melspec(audio)
            else:
                mel1 = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                 self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                 center=False)
            
            mel = mel.permute(0, 2, 1) # (1, dim, seq_len) --> (1, seq_len, dim)
        else:
            mel = torch.load(row.feat_path, map_location='cpu').float() 

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0) # (1, seq_len, dim)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(1) - frames_per_seg - 1)
                    mel = mel[:, mel_start:mel_start + frames_per_seg, :]
                    audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                else:
                    mel = torch.nn.functional.pad(mel, (0, 0, 0, frames_per_seg - mel.size(2)), 'constant')
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')


        if self.use_alt_melcalc:
            mel_loss = self.alt_melspec(audio)
        else:
            mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)
        return (mel.squeeze(), audio.squeeze(0), str(row.audio_path), mel_loss.squeeze())

    def __len__(self):
        return len(self.audio_files)
