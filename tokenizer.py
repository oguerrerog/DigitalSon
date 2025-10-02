import os
from transformers import AutoTokenizer

def load_tokenizer(model_name="distilgpt2"):
    """
    Load HF tokenizer, set pad token if missing.
    The tokenizer is cached by HF; repeated calls will reuse cache.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
