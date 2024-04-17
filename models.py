from datasets import load_dataset
import openai
import os
import itertools
import logging
import string
import sys

import dspy
from dsp.utils import deduplicate
from dspy.teleprompt import BootstrapFewShot, LabeledFewShot, BayesianSignatureOptimizer, BootstrapFewShotWithRandomSearch
from dspy.evaluate import answer_exact_match, answer_passage_match
from dspy.evaluate.evaluate import Evaluate
from dsp.utils import deduplicate
from dspy.primitives import module

from copy import copy
import random
import json
import tqdm
import pickle
import pandas as pd
from pandas import json_normalize
import numpy as np

random.seed(1)
logging.basicConfig(level=logging.DEBUG)
# Signatures

class VerifyQAAnswer(dspy.Signature):
    # __doc__ = """Verify if the response to the question is aligned with the context."""
    __doc__ = """You are a hallucination checking agent. Verify if the given response to the question can be derived from the provided context."""
    context = dspy.InputField(desc="may contain relevant facts")
    response = dspy.InputField()
    question = dspy.InputField(desc="original question that needs to be answered")
    answer = dspy.OutputField(desc="Return a binary response, yes or no")

class ContextQASignature(dspy.Signature):
    __doc__ = """Answer questions with short factoid answers."""
    context = dspy.InputField(desc="contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc = 'Do not repeat the question, only show final answer.')

class ModifySignature(dspy.Signature):
    __doc__ = """The response to the question is wrong. Generate a correct answer to the question that is different from the given response and aligns with the provided context."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    response = dspy.InputField()
    answer = dspy.OutputField(desc = 'Based on the context and taking the feedback on the response to the question into consideration, answer the question with short answers (limited to less than 6 words)')

# Models

class HaluCheckRAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(VerifyQAAnswer)

    def forward(self, question, response, context=None):
        if context is None:
            context = self.retrieve(question).passages
        prediction = self.generate_answer(question=question, response=response, context=context)
        response = prediction.answer.lower().strip(string.punctuation)
        if response not in ["yes", "no"]:
            self.logger.error(f"Response to {question=} not as expected, cannot decide on hallucination")
            return None
        if response == "yes":
            return False
        return True 
    
class CloseQARAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.Predict(ContextQASignature)

    def forward(self, context, question):
        prediction = self.generate_answer(context = context, question=question)
        return prediction.answer
    
class OpenQARAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.Predict(ContextQASignature)

    def forward(self, question, context=None):
        if context is None:
            context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return prediction.answer
        
class ModifyOpenQARAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.Predict(ModifySignature)

    def forward(self, question, response, context=None):
        if context is None:
            context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question, response=response)
        return prediction.answer

class MitigationOpenAQRAG(dspy.Module):
    def __init__(self, num_passages=5):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.openqa = OpenQARAG()
        self.halu_check = HaluCheckRAG()
        self.modify = ModifyOpenQARAG()
        self.logger = logging.getLogger(__name__)

    def forward(self, question):
        context = self.retrieve(question).passages
        initial_response = self.openqa(question=question, context=context)
        hallucination = self.halu_check(question=question, response=initial_response, context=context)
        modified_response = None
        if hallucination:
            modified_response = self.modify(question=question, response=initial_response, context=context)
        self.logger.debug(f'{initial_response=}')
        self.logger.debug(f'{hallucination=}')
        self.logger.debug(f'{modified_response=}')
        return initial_response, hallucination, modified_response