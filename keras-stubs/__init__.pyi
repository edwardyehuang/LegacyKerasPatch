"""Type stubs for keras module with ops support from LegacyKerasPatch."""

from typing import Any

from keras import ops as ops
from keras import activations as activations
from keras import applications as applications
from keras import backend as backend
from keras import callbacks as callbacks
from keras import config as config
from keras import constraints as constraints
from keras import datasets as datasets
from keras import distribution as distribution
from keras import initializers as initializers
from keras import layers as layers
from keras import losses as losses
from keras import metrics as metrics
from keras import mixed_precision as mixed_precision
from keras import models as models
from keras import optimizers as optimizers
from keras import preprocessing as preprocessing
from keras import random as random
from keras import regularizers as regularizers
from keras import saving as saving
from keras import utils as utils

def __getattr__(name: str) -> Any: ...
