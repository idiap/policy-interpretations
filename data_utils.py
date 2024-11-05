# SPDX-FileCopyrightText: 2024 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" Data utils for preprocessing. """

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class BartData:

    def __init__(self, tokenizer, separate_sentences=False, max_src_tokens=1024, max_tgt_tokens=512):
        self.tokenizer = tokenizer
        self.separate_sentences = separate_sentences
        self.max_src_tokens = max_src_tokens
        self.max_tgt_tokens = max_tgt_tokens

    def encode(self, example):
        if self.separate_sentences:
            src = [ids for sent in example['src_sents'] for ids in self.tokenizer.encode(sent)]
            if len(src) > self.max_src_tokens:
                src = src[:self.max_src_tokens - 1] + [self.tokenizer.eos_token_id]
        else:
            src = self.tokenizer.encode(
                ' '.join(example['src_sents']), max_length=self.max_src_tokens, truncation=True
            )
        tgt = self.tokenizer.encode(example['tgt'], max_length=self.max_tgt_tokens - 1, truncation=True)
        tgt = [self.tokenizer.eos_token_id] + tgt  # prepend EOS token (BART format)
        return {
            'src': src,
            'tgt': tgt,
            'name': example['name'],
            'tgt_i': example['tgt_i'],
        }
