import librosa
import numpy as np

sample_rate=16000
n_mels=80
n_fft=1024
hop_length=512

def wav_to_feature(wav_paths, store=False):
    feature_list = []
    for wav_path in wav_paths:
        # 1. load wav and trim silence
        try:
            wav, _ = librosa.load(wav_path, sr=sample_rate)
            wav = trim_silence(wav)
        except ValueError:
            print("can't load/trim file: {}. Skip it".format(wav_path))
            continue

        # 2. Remove some too short wav and do Normalization.
        if len(wav) < (sample_rate//2):
            print("File: {} are too short (< 0.5s). Skip it".format(wav_path))
            continue
        max_value = np.abs(wav).max()
        if max_value == 0.0:
            print("File: {} containszero samples only. Skip it".format(wav_path))
            continue
        wav = wav/max_value
        
        # 3. Load Feature (mel and mfcc), concatenate them 
        # into feature (80+20, frame_num)
        mel = get_mel(wav)
        mfcc = get_mfcc(wav)
        feature = np.concatenate([mel, mfcc], axis=0)

        # 4. Either we return the feature list directly, 
        # or we store the feature and return the path list.
        if store:
            feature_path = wav_path.replace('.wav', '.npy')
            np.save(feature_path, feature)
            feature_list.append(feature_path)
        else:
            feature_list.append(feature)


    return feature_list


def get_mel(wav):
    melspec = librosa.feature.melspectrogram(wav,sr=sample_rate, 
            n_mels=n_mels,n_fft=n_fft, hop_length=hop_length)
    return melspec

def get_mfcc(wav):
    mfcc = librosa.feature.mfcc(wav,sr=sample_rate, 
            n_mels=n_mels,n_fft=n_fft, hop_length=hop_length)

    return mfcc

def trim_silence(wav):
    _, ints = librosa.effects.trim(wav, top_db=50, frame_length=256, hop_length=64)
    start = int(max(ints[0]-sample_rate*0.2, 0))
    end   = int(min(ints[1]+sample_rate*0.2, len(wav)))
    return wav[start:end]
