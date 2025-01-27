def deterministic_run(seed:int, torch_det_algos: bool = False):
    import random
    random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch_det_algos:
        torch.use_deterministic_algorithms(mode=True) # will throw excep. when trying to use non-det. algos
    import numpy as np
    np.random.seed(seed)