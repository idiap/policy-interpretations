# SPDX-FileCopyrightText: 2024 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" T5 model family. """

from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

from transformers import T5ForConditionalGeneration, T5Tokenizer

from bart import BartSummarizer


class T5Summarizer(BartSummarizer):

    def load_model(self, model_name_or_path, special_tokens):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        self.model.resize_token_embeddings(len(self.tokenizer))  # extend embedding matrices for special tokens

    def configure_optimizers(self):
        # remove shared embedding matrix (used in LM head and can't have the same params in different param groups)
        encoder_params = (p for n, p in self.model.encoder.named_parameters() if n != 'embed_tokens.weight')  # T5 uses model.encoder
        decoder_params = (p for n, p in self.model.decoder.named_parameters() if n != 'embed_tokens.weight')  # T5 uses model.decoder
        optimizer = Adam([
            {'params': encoder_params},
            {'params': decoder_params},
            {'params': self.model.lm_head.parameters()},
        ])
        scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=[self.hparams.max_lr_enc, self.hparams.max_lr_dec, self.hparams.max_lr_lm_head],
            total_steps=self.hparams.max_steps,
            pct_start=self.hparams.warmup,
            anneal_strategy=self.hparams.lr_anneal,
            cycle_momentum=False,
            div_factor=self.hparams.div_warmup,
            final_div_factor=self.hparams.div_final,
            last_epoch=-1,  # TODO: enable resume training
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]

    def decode(self, output_ids):
        """ Decode BPE tokens into text. """
        text = self.tokenizer.decode(output_ids, spaces_between_special_tokens=True)
        stst_start = self.annotation_schema.mapping['standardized sentence']['text_start']
        stst_end = self.annotation_schema.mapping['standardized sentence']['text_end']
        text = text.replace(f'{stst_end} {stst_start}', f'{stst_end}<sent>{stst_start}')
        text = text.replace(self.tokenizer.pad_token, '')  # T5 uses pad instead of BOS token
        text = text.replace(self.tokenizer.eos_token, '')
        text = ' '.join(text.split())
        return text
