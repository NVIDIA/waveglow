from .inference import synthesize
import torch


def test_synthesize():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mel = torch.randn(2, 80, 10)
    audio = synthesize(mel, 'cpu')
    assert list(audio.size()) == [2, 2560], 'the size of the audio should be {}'.format([2, 2560])
