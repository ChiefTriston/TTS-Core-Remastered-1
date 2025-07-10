# 5_sde_refiner/blocks/utils.py

def assert_shape(x, expected):
    """
    Simple utility to check a tensor shape at runtime. 
    (Call it if you want to sanity‐check intermediate sizes.)
    """
    if tuple(x.shape) != tuple(expected):
        raise RuntimeError(f"Shape mismatch: got {tuple(x.shape)}, expected {expected}")

def profile(model, *args, **kwargs):
    """
    Placeholder profiling helper—can be filled in later.
    """
    return model(*args, **kwargs)
