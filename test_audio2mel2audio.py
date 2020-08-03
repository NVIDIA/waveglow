from waveglow.layers import TacotronSTFT
import librosa
import torch
from waveglow import  synthesize


f = librosa.util.example_audio_file()
y, data = librosa.load(f, sr=22050)
y = torch.from_numpy(y).unsqueeze(0)

stft = TacotronSTFT()
mel = stft.mel_spectrogram(y)
print(mel.size())


wav = synthesize(mel, is_fp16=True)
print(wav.size())
