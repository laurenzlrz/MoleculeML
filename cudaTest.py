import torch.cuda as cuda
import torch

print(cuda.is_available())
print("Torch Version:", torch.__version__)
print("CUDA Version (falls verfügbar):", torch.version.cuda)
print(torch.cuda.current_device())