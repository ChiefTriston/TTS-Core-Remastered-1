import torch
from .config import RefEncConfig
from .encoder import ReferenceEncoder

if __name__ == '__main__':
    # Initialize configuration and model
    cfg = RefEncConfig()
    model = ReferenceEncoder(cfg).eval()

    # Load best checkpoint
    ckpt = torch.load('best_model.pt', map_location='cpu')
    model.load_state_dict(ckpt['model'])

    # 1) Save a scripted (TorchScript) version
    scripted = torch.jit.script(model)
    scripted.save('reference_encoder_scripted.pt')
    print("✔ Saved scripted model to reference_encoder_scripted.pt")

    # 2) Optionally fuse Conv+BN layers via FX
    try:
        import torch.fx as fx
        from torch.fx.passes.fuser import fuse

        graph = fx.symbolic_trace(model)
        fused = fuse(graph)
        torch.jit.save(torch.jit.script(fused), 'reference_encoder_fused.pt')
        print("✔ Saved FX-fused model to reference_encoder_fused.pt")
    except Exception as e:
        print(f"⚠ FX fusion not performed: {e}")