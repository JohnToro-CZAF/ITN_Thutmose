from abc import ABC
from typing import List, Union

import torch
from pytorch_lightning.core.module import _jit_is_scripting
from torch.onnx import TrainingMode

class Exportable(ABC):
    @property
    def input_module(self):
        return self
    
    @property
    def output_module(self):
        return self
    
    def export(
        
    )