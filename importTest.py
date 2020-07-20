import numpy as np 
import matplotlib.pyplot as plt

#import cntk as C
from cntk.learners import learning_rate_schedule, UnitType

import simulation
import cntk_unet

from cntk.device import try_set_default_device, gpu, get_gpu_properties

import cntk as C
from cntk.layers import Convolution, MaxPooling, Dense
from cntk.initializer import glorot_uniform
from cntk.ops import relu, sigmoid, input_variable

import random