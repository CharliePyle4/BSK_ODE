import os
os.environ["KERAS_BACKEND"] = "torch"  # before importing keras_sig / keras


import torch
torch.set_default_dtype(torch.float64)
active_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", active_device, "torch", torch.__version__)


import time
import math
import numpy as np
import torchmin
from torch import nn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from fbm import FBM
from torchmin import least_squares
from torchmin import minimize
from torch import cumulative_trapezoid  # or torch.cumulative_trapezoid in newer versions


import keras_sig
from keras_sig import SigLayer

import math
from typing import Tuple
import torch
from fbm import FBM


# Cell 3 - seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False