"""
Mostly taken from facebook
 Command line tool to get dense results and validate them
"""


import argparse
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
from torch import Tensor as Tensr
from torch import nn

from dpr.data.qa_validation import calculate_matches
from dpr.models import init_biencoder_components
from dpr.options import add_encoder_params, setup_args_gpu, print_args, set_encoder_params_from_state, \
    add_tokenizer_params, add_cuda_params
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import setup_for_distributed_mode, get_model_obj, load_states_from_checkpoint
from dpr.indexer.faiss_indexers import DenseIndexer, DenseHNSWFlatIndexer, DenseFlatIndexer

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

# Asai has the question vector generator as a function inside the class facebooks has it as a class indepentent function
class DenseRetriever(object):
    """
    Retrieves Passages over the provided index and question encoder    
    """
    def __init__(self, question_encoder: nn.Module, batch_size: int, tensorizer: Tensorizer, index: DenseIndexer):
        self.question_encoder = question_encoder
        self.batch_size = batch_size
        self.tensorizer = tensorizer
        self.index = index

    def generate_question_vectors(self, questions: List[str]) -> Tensr:
        """
        
        """
        num_questions = len(questions)
        batch_sz = self.batch_size
        query_vectors = []

        with torch.no_grad():  # turns off autograd
            for j, batch_start in enumerate(range(0, num_questions, batch_sz)):

                batch_token_tensors = [self.tensorizer.text_to_tensor(q) for q in questions[batch_start:batch_start + batch_sz]] # not implemented?

                q_ids_batch = torch.stack(batch_token_tensors, dim=0) # concatenates sequence of tensors
                q_seg_batch = torch.zeros_like(q_ids_batch) # tensor of 0s same size as input (q_ids_batch)
                q_attn_mask = self.tensorizer.get_attn_mask(q_ids_batch) # not implemented?
                _, out, _ = self.question_encoder(q_ids_batch, q_seg_batch, q_attn_mask)

                query_vectors.extend(out.cpu().split(1, dim=0))

                if len(query_vectors) % 100 == 0:
                    logger.info('Encoded queries %d', len(query_vectors))
                    
            query_tensor = torch.cat(query_vectors, dim=0)

            logger.info('Total encoded queries tensor %s', query_tensor.size())

            assert query_tensor.size(0) == len(questions)
            return query_tensor

    def index_encoded_data(self, vector_files: List[str], buffer_size: int = 5000):
         """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        time0 = time.time()
        results = self.index.search_knn(query_vectors, top_docs)
        logger.info('index search time: %f sec.', time.time() - time0)
        return results