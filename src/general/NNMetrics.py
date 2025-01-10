from enum import Enum

class NNMetrics(Enum):
    MSE = "MSE"
    MAE = "MAE"
    R2 = "R2"
    NRMSE = "NRMSE"
    EPOCH_TIME = "epoch_time"
    EPOCH_MEMORY = "epoch_memory"
    EPOCH_GPU_MEMORY = "epoch_gpu_memory"

