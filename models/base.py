import copy
import hashlib
import json
import os
from typing import Any, Optional

from omegaconf import DictConfig, Omegaconf
from pytorch_lightning import Trainer
from pytorch_lightning.core.saving import _load_state as ptl_load_state
from pytorch_lightning.core.saving import load_hparams_from_tags_csv, load_hparams_from_yaml
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.migration import pl_legacy_patch
from transformers import TRANSFORMERS_CACHE

# ModelPT.
from .modelpt import ModelPT

# TODO: Exportable.
# This interface should be implemented by particular classes derived from nemo.core.NeuralModule or
# nemo.core.ModelPT it gives these entities ability to be exported for deployment to formats such as ONNX.
# from .exportable import Exportable

__all__ = ["NLPModel"]

NEMO_NLP_TMP = os.path.join(os.path.dirname(str(TRANSFORMERS_CACHE)), "nemo_nlp_tmp")

os.makedirs(NEMO_NLP_TMP, exist_ok=True)

class NLPModel(ModelTP):
    "Base class for NLP Models".
    