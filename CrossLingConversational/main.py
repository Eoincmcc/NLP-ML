## Copying Cora

# Data Download
## run install-script.py - working

# Train retriever/retriever
## Use only subsection of data to run locally

import argparse
from msilib.schema import Class
import os
import csv
import glob
import json
import gzip
import logging
import pickle
import time
import jsonlines
from typing import List, Tuple, Dict, Iterator

import numpy as np
import torch
from torch import Tensor as Tensor
from torch import nn

# import calculate matches
# import biencoder components
# paramters and setup arguments
# tensorizer
# distributed mode setup
# facebook indexers



def logger_start():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if(logger.hasHandlers()):
        logger.handlers.clear()
    console = logging.StreamHandler()
    logger.addHandler(console)

logger_start()

class DenseRetriever(object):
    # Passage Retrieving over the provided index and question encoder.

    def __init__(self, question_encoder: nn.Module, batch_size: int, tensorizer: Tensorizer, 
        pass


# Convert Data
# Run Generator
# Run Evaluation



