class AudioFeature:
    import os 
    import numpy as np
    import librosa
    import pandas as pd
    
    def __init__(self, path):
        self.path = path

        if os.path.exists(self.path):
            self.y, self.sr = librosa.load(self.path)

        else:
            raise Exception(f"Path not found - {self.path}")

        self.load_audio_file()


    def get_dataframe(self):
        """
        Return all extracted features as a single-row pandas DataFrame
        with columns matching the feature names.
        """
        features = self.get_all_params()
        
        df = pd.DataFrame([features])
        
        return df

    def get_all_params(self):

        self.get_length()
        self.get_zero_crossings()
        self.get_tempo()
        self.get_centroids()
        self.get_spectral_rolloff()
        self.get_mel_frequencies()
        self.get_chroma()
        self.get_rms()
        self.get_spectral_bandwith()
        self.get_harmony()
        self.get_perceptr()
        
        result = {
            "length" : self.length,
            "chroma_stft_mean" : self.chroma_stft_mean,
            "chroma_stft_var" : self.chroma_stft_var,
            "rms_mean" : self.rms_mean,
            "rms_var" : self.rms_var,
            "spectral_centroid_mean" : self.spectral_centriod_mean,
            "spectral_centroid_var" : self.spectral_centriod_var,
            "spectral_bandwith_mean" : self.spectral_bandwith_mean,
            "spectral_bandwith_var" : self.spectral_bandwith_var,
            "rolloff_mean" : self.rolloff_mean,
            "rolloff_var" : self.rolloff_var,
            "zero_crossing_rate_mean": self.zero_crossings_rate_mean,
            "zero_crossing_rate_var" : self.zero_crossings_rate_var,
            "harmony_mean" : self.harmony_mean,
            "harmony_var" : self.harmony_var,
            "perceptr_mean" : self.perceptr_mean,
            "perceptr_var": self.perceptr_var,
            "tempo" : self.tempo

        }

        result.update(self.mel_frequencies)

        return  result


    def load_audio_file(self):
        self.audio_file, _ = librosa.effects.trim(self.y)

        return self.audio_file

    def get_harmony(self):
        harmony = librosa.effects.harmonic(y = self.audio_file)
        self.harmony_mean = np.mean(harmony)
        self.harmony_var = np.var(harmony)

        return self.harmony_mean, self.harmony_var
        

    def get_length(self):
        self.length = np.shape(self.audio_file)[0]
        
        return self.length

    def get_zero_crossings(self):
        zero_crossings = librosa.zero_crossings(self.audio_file, pad=False)

        self.zero_crossings_rate_mean = np.mean(zero_crossings)
        self.zero_crossings_rate_var = np.var(zero_crossings)

        return self.zero_crossings_rate_mean, self.zero_crossings_rate_var

    def get_rms(self):
        rms = librosa.feature.rms(y = self.audio_file)

        self.rms_mean = np.mean(rms)
        self.rms_var = np.var(rms)

        return self.rms_mean, self.rms_var

    def get_spectral_bandwith(self):
        spectral_bandwith = librosa.feature.spectral_bandwidth(y = self.audio_file, sr = self.sr)
        self.spectral_bandwith_mean = np.mean(spectral_bandwith)
        self.spectral_bandwith_var = np.var(spectral_bandwith)

        return self.spectral_bandwith_mean, self.spectral_bandwith_var
    

    def get_tempo(self):
        # Estimate tempo (BPM) from your audio time series
        self.tempo, _ = librosa.beat.beat_track(
            y=self.y,
            sr=self.sr
        )

        self.tempo = self.tempo[0]
        
        return self.tempo

    def get_centroids(self):
        spectral_centroids = librosa.feature.spectral_centroid(y = self.audio_file, sr=self.sr)[0]

        self.spectral_centriod_mean = np.mean(spectral_centroids)
        self.spectral_centriod_var = np.var(spectral_centroids)

        return self.spectral_centriod_mean, self.spectral_centriod_var

    def get_perceptr(self):
        # 1. Compute Mel‐spectrogram
        S = librosa.feature.melspectrogram(
            y=self.audio_file,
            sr=self.sr,
            hop_length=5000,
        )
        
        pcen = librosa.pcen(
            S,
            sr=self.sr,
            hop_length=5000,
        )
        
        # 3. Compute statistics
        self.perceptr_mean = np.mean(pcen)
        self.perceptr_var  = np.var(pcen)
        
        
        return pcen
    
    def get_spectral_rolloff(self):

        spectral_rolloff = librosa.feature.spectral_rolloff(y = self.audio_file, sr=self.sr)[0]

        self.rolloff_mean = np.mean(spectral_rolloff)
        self.rolloff_var = np.var(spectral_rolloff)


        return self.rolloff_mean, self.rolloff_var

    def get_chroma(self, hop_length = 5000):
        
        chromagram = librosa.feature.chroma_stft(y = self.audio_file, sr=self.sr, hop_length=hop_length)
        self.chroma_stft_mean = np.mean(chromagram)
        self.chroma_stft_var = np.var(chromagram)

        return self.chroma_stft_mean , self.chroma_stft_var
    
    def get_mel_frequencies(self):
        mfccs = librosa.feature.mfcc(y = self.audio_file, sr=self.sr)
        self.mel_frequencies = {}
        for i, mfcc in enumerate( mfccs):
            self.mel_frequencies[f'mfcc{i+1}_mean'] = np.mean(mfcc)
            self.mel_frequencies[f'mfcc{i+1}_var'] = np.var(mfcc)

        return self.mel_frequencies
        