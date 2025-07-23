# train/adapters/vocoder_adapter.py
import torch
from vocoder.trainer import VocoderTrainer

def vocoder_adapter(trainer: VocoderTrainer):
    """Adapts VocoderTrainer to TrainBlock interface."""
    class VocoderAdapter:
        def __init__(self, t: VocoderTrainer):
            self.generator = t.generator
            self.discriminators = t.discriminators
            self._t = t

        def compute_discriminator_loss(self, fake, real):
            if hasattr(self._t, "compute_discriminator_loss"):
                return self._t.compute_discriminator_loss(fake, real)
            return torch.nn.functional.binary_cross_entropy(fake, torch.zeros_like(fake)) + \
                   torch.nn.functional.binary_cross_entropy(real, torch.ones_like(real))

        def compute_generator_loss(self, wav_fake, wav_gt, disc_out):
            if hasattr(self._t, "compute_generator_loss"):
                result = self._t.compute_generator_loss(wav_fake, wav_gt, disc_out)
                if isinstance(result, torch.Tensor):
                    return result, torch.tensor(0.0, device=wav_fake.device), torch.tensor(0.0, device=wav_fake.device)
                return result
            g_loss = torch.nn.functional.binary_cross_entropy(disc_out, torch.ones_like(disc_out))
            stft_loss = torch.nn.functional.l1_loss(wav_fake, wav_gt)
            return g_loss, stft_loss, torch.tensor(0.0, device=wav_fake.device)

        def compute_discriminator_metrics(self, fake_out, real_out):
            if hasattr(self._t, "compute_discriminator_metrics"):
                m = self._t.compute_discriminator_metrics(fake_out, real_out)
                return {k: float(v) for k, v in m.items()}
            return {"d_acc": float(((fake_out < 0.5).float().mean() + (real_out > 0.5).float().mean()) / 2)}
    
    return VocoderAdapter(trainer)