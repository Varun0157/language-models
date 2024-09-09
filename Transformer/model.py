# This programming exercise will give you the chance to use the Pytorch to implement a language
# model in accordance with the Transformer Decoder architecture. Modern language models, such as
# OpenAI's GPT series, rely heavily on the Transformer Decoder. In this assignment, you will implement
# the core components of the Transformer Decoder and train it to generate coherent and contextually
# relevant text. The task at hand involves constructing a language model with the capability to predict
# the subsequent word in a given sequence of words. The model will be trained on a text corpus ,
# which then will learn to produce text that mirrors the structure and pattern of the training data. 


import torch
import torch.nn as nn



