import torch
from torch.hub import load_state_dict_from_url

from .glow import WaveGlow

_model = None

_model_url = 'https://www.dropbox.com/s/lpwq2mcodmeuoay/waveglow_256channels_universal_v5.pth?dl=1'

_model_config = dict(n_mel_channels=80,
                     n_flows=12,
                     n_group=8,
                     n_early_every=4,
                     n_early_size=2,
                     WN_config=dict(n_layers=8, n_channels=256, kernel_size=3))


@torch.no_grad()
def synthesize(mel: torch.Tensor,
               device='cuda',
               is_fp16: bool = False,
               sigma: float = 1.0) -> torch.Tensor:
    global _model
    if _model is None:
        _model = WaveGlow(**_model_config)
        state_dict = load_state_dict_from_url(_model_url, progress=True)
        _model.load_state_dict(state_dict)
        WaveGlow.remove_weightnorm(_model)

        _model.eval()
        _model.to(device)

        if is_fp16:
            from apex import amp
            _model, _ = amp.initialize(_model, [], opt_level="O3")

    mel = mel.to(device)
    mel = mel.half() if is_fp16 else mel

    audio = _model.infer(mel, sigma=sigma)

    return audio.cpu()
