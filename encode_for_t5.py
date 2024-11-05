# SPDX-FileCopyrightText: 2024 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" Encode text data for T5 generation. """

import json
import os
import torch
import tqdm
from transformers import T5Tokenizer, AddedToken

from data_schema import SchemaFactory


SPLIT_SYMBOL = '<sent>'


class T5Data:

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
        tgt = [self.tokenizer.pad_token_id] + tgt  # prepend <pad> token (T5 format)
        return {
            'src': src,
            'tgt': tgt,
            'name': example['name'],
            'tgt_i': example['tgt_i'],
        }


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    special_tokens = SchemaFactory.get_schema(args.dataset).get_special_text_tokens()
    special_tokens = [AddedToken(t) for t in special_tokens]
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name, additional_special_tokens=special_tokens)
    encoder = T5Data(tokenizer, args.separate_sentences, args.max_src_tokens, args.max_tgt_tokens)
    for filter_model in ['filterbert', 'oracle', 'lead']:
        for split in ['train', 'valid', 'test']:
            data_path = os.path.join(args.text_dir, f'{args.dataset}.{filter_model}.{split}.json')
            if os.path.exists(data_path):
                with open(data_path, 'r') as f:
                    data = json.load(f)
                outputs = []
                for example in tqdm.tqdm(data):
                    example['tgt'] = example['tgt'].replace(SPLIT_SYMBOL, ' ')
                    outputs.append(encoder.encode(example))
                torch.save(outputs, os.path.join(args.output_dir, f'{args.dataset}.{filter_model}.{split}.pt'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Encode text for T5 generation.')
    parser.add_argument('--tokenizer_name', default='google-t5/t5-base', help='Tokenizer name or path to dir')
    parser.add_argument('--dataset', default='us-russia', choices=['us-russia'], help='Dataset name')
    parser.add_argument('--separate_sentences', action='store_true',
                        help='Encode source sentences separately, surrounded by BOS/EOS tokens.')
    parser.add_argument('--max_src_tokens', type=int, default=1024, help='Maximum number of source tokens')
    parser.add_argument('--max_tgt_tokens', type=int, default=512, help='Maximum number of target tokens')
    parser.add_argument('--text_dir', required=True, help='Path to text dir')
    parser.add_argument('--output_dir', required=True, help='Path to output dir')
    main(parser.parse_args())
