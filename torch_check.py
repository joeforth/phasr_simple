import torch
print(torch.__version__)            # look at the suffix
print(torch.version.cuda)           # None  = CPU-only build
print(torch.cuda.is_available())    # the headline answer
print(torch.cuda.device_count())