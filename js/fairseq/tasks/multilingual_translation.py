# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import OrderedDict
import copy
import os

import torch

from torch.autograd import Variable

from fairseq import options, utils
from fairseq.data import (
    Dictionary,
    LanguagePairDataset,
    RoundRobinZipDatasets,
    TransformEosLangPairDataset,
    indexed_dataset,
)
from fairseq.models import FairseqMultiModel
from fairseq.tasks.translation import load_langpair_dataset


from . import FairseqTask, register_task


def _lang_token(lang: str):
    return f'__{lang}__'


def _lang_token_index(dic: Dictionary, lang: str):
    """Return language token index."""
    idx = dic.index(_lang_token(lang))
    assert idx != dic.unk_index, \
        f'cannot find language token for lang {lang}'
    return idx


@register_task('multilingual_translation')
class MultilingualTranslationTask(FairseqTask):
    """A task for training multiple translation models simultaneously.

    We iterate round-robin over batches from multiple language pairs, ordered
    according to the `--lang-pairs` argument.

    The training loop is roughly:

        for i in range(len(epoch)):
            for lang_pair in args.lang_pairs:
                batch = next_batch_for_lang_pair(lang_pair)
                loss = criterion(model_for_lang_pair(lang_pair), batch)
                loss.backward()
            optimizer.step()

    In practice, `next_batch_for_lang_pair` is abstracted in a FairseqDataset
    (e.g., `RoundRobinZipDatasets`) and `model_for_lang_pair` is a model that
    implements the `FairseqMultiModel` interface.

    During inference it is required to specify a single `--source-lang` and
    `--target-lang`, instead of `--lang-pairs`.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', metavar='DIR', help='path to data directory')
        parser.add_argument('--lang-pairs', default=None, metavar='PAIRS',
                            help='comma-separated list of language pairs (in training order): en-de,en-fr,de-fr')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language (only needed for inference)')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language (only needed for inference)')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', default=False, action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left (default: True)')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left (default: False)')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--encoder-langtok', default=None, type=str, choices=['src', 'tgt'],
                            metavar='SRCTGT',
                            help='replace beginning-of-sentence in source sentence with source or target '
                                 'language token. (src/tgt)')
        parser.add_argument('--decoder-langtok', action='store_true',
                            help='replace beginning-of-sentence in target sentence with target language token')
        # fmt: on

    def __init__(self, args, dicts, training):
        super().__init__(args)
        self.dicts = dicts
        

        self.lang_pairs = args.lang_pairs
        # eval_lang_pairs for multilingual translation is usually all of the
        # lang_pairs. However for other multitask settings or when we want to
        # optimize for certain languages we want to use a different subset. Thus
        # the eval_lang_pairs class variable is provided for classes that extend
        # this class.
        self.eval_lang_pairs = args.lang_pairs
        # model_lang_pairs will be used to build encoder-decoder model pairs in
        # models.build_model(). This allows multitask type of sub-class can
        # build models other than the input lang_pairs
        self.model_lang_pairs = copy.copy(args.lang_pairs)
        self.langs = list(dicts.keys())
        self.training = training

    @classmethod
    def setup_task(cls, args, **kwargs):
        dicts, training = cls.prepare(args, **kwargs)
        return cls(args, dicts, training)

    @classmethod
    def prepare(cls, args, **kargs):
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        if getattr(args, 'raw_text', False):
            utils.deprecation_warning('--raw-text is deprecated, please use --dataset-impl=raw')
            args.dataset_impl = 'raw'
        elif getattr(args, 'lazy_load', False):
            utils.deprecation_warning('--lazy-load is deprecated, please use --dataset-impl=lazy')
            args.dataset_impl = 'lazy'

        args.lang_pairs = args.lang_pairs.split(',')
        sorted_langs = sorted(list({x for lang_pair in args.lang_pairs for x in lang_pair.split('-')}))
        if args.source_lang is not None or args.target_lang is not None:
            training = False
            args.lang_pairs = ['{}-{}'.format(args.source_lang, args.target_lang)]
        else:
            training = True
            args.source_lang, args.target_lang = args.lang_pairs[0].split('-')

        # load dictionaries
        dicts = OrderedDict()
        for lang in sorted_langs:
            paths = args.data.split(':')
            assert len(paths) > 0
            dicts[lang] = Dictionary.load(os.path.join(paths[0], 'dict.{}.txt'.format(lang)))
            if len(dicts) > 0:
                assert dicts[lang].pad() == dicts[sorted_langs[0]].pad()
                assert dicts[lang].eos() == dicts[sorted_langs[0]].eos()
                assert dicts[lang].unk() == dicts[sorted_langs[0]].unk()
            if args.encoder_langtok is not None or args.decoder_langtok:
                for lang_to_add in sorted_langs:
                    dicts[lang].add_symbol(_lang_token(lang_to_add))
            print('| [{}] dictionary: {} types'.format(lang, len(dicts[lang])))
        return dicts, training

    def get_encoder_langtok(self, src_lang, tgt_lang):
        if self.args.encoder_langtok is None:
            return self.dicts[src_lang].eos()
        if self.args.encoder_langtok == 'src':
            return _lang_token_index(self.dicts[src_lang], src_lang)
        else:
            return _lang_token_index(self.dicts[src_lang], tgt_lang)

    def get_decoder_langtok(self, tgt_lang):
        if not self.args.decoder_langtok:
            return self.dicts[tgt_lang].eos()
        return _lang_token_index(self.dicts[tgt_lang], tgt_lang)

    def alter_dataset_langtok(self, lang_pair_dataset,
                              src_eos=None, src_lang=None, tgt_eos=None, tgt_lang=None):
        if self.args.encoder_langtok is None and not self.args.decoder_langtok:
            return lang_pair_dataset

        new_src_eos = None
        if self.args.encoder_langtok is not None and src_eos is not None \
           and src_lang is not None and tgt_lang is not None:
            new_src_eos = self.get_encoder_langtok(src_lang, tgt_lang)
        else:
            src_eos = None

        new_tgt_bos = None
        if self.args.decoder_langtok and tgt_eos is not None and tgt_lang is not None:
            new_tgt_bos = self.get_decoder_langtok(tgt_lang)
        else:
            tgt_eos = None

        return TransformEosLangPairDataset(
            lang_pair_dataset,
            src_eos=src_eos,
            new_src_eos=new_src_eos,
            tgt_bos=tgt_eos,
            new_tgt_bos=new_tgt_bos,
        )

    def load_dataset(self, split, epoch=0, **kwargs):
        """Load a dataset split."""

        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]
        # import sys
        # print("split {}".format(split))
        # print("args.lang_pairs {}".format(self.args.lang_pairs))
        # sys.exit(0)

        def language_pair_dataset(lang_pair):
            src, tgt = lang_pair.split('-')
            langpair_dataset = load_langpair_dataset(
                data_path, split, src, self.dicts[src], tgt, self.dicts[tgt],
                combine=True, dataset_impl=self.args.dataset_impl,
                upsample_primary=self.args.upsample_primary,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
            )
            return self.alter_dataset_langtok(
                langpair_dataset,
                src_eos=self.dicts[tgt].eos(),
                src_lang=src,
                tgt_lang=tgt,
            )

        self.datasets[split] = RoundRobinZipDatasets(
            OrderedDict([
                (lang_pair, language_pair_dataset(lang_pair))
                for lang_pair in self.args.lang_pairs
            ]),
            eval_key=None if self.training else "%s-%s" % (self.args.source_lang, self.args.target_lang),
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        lang_pair = "%s-%s" % (self.args.source_lang, self.args.target_lang)
        return RoundRobinZipDatasets(
            OrderedDict([(
                lang_pair,
                self.alter_dataset_langtok(
                    LanguagePairDataset(
                        src_tokens, src_lengths,
                        self.source_dictionary
                    ),
                    src_eos=self.source_dictionary.eos(),
                    src_lang=self.args.source_lang,
                    tgt_lang=self.args.target_lang,
                ),
            )]),
            eval_key=lang_pair,
        )

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)
        if not isinstance(model, FairseqMultiModel):
            raise ValueError('MultilingualTranslationTask requires a FairseqMultiModel architecture')
        return model


    def denoise_tensor (self, x, ratio, window):

        def drop_elements(x, ratio):
            u = torch.randperm(x.size(0)-1)
            k = int(x.size(0) * ratio)
            ids = u[:k]
            x[ids] = -100
            # x[ids] = 996 # "_"

            return x[x!=-100]
            # return x


        def permute(x, k):
            ulist = []
            for i in range(0,x.size(0)-1,k):
                g = x[i:min(i+k,x.size(0)-1)]
                u = torch.randperm(g.size(0))
                ulist.append(g[u])
            ret = torch.cat(ulist+[x[-1:]],0)

            return ret

        a_stack = []

        for i in range(x.size(0)):
            uu = drop_elements(x[i], ratio) if ratio > 0.0 else x[i]
            uuu = permute(uu, window) if window > 0 else uu
            a_stack.append(uuu)

        res = torch.stack(a_stack,dim=0)

        return res


    def train_step(self, epch, step, sample, model, criterion, optimizer, dmodel, generator,ignore_grad=False):
        model.train()

        # print("count {}".format(step))
        #
        # Lample, et al. (2018) Unsupervised Machine Translation using Monolingual Corpora only. ICLR 2018.
        #

        def denoise_input (lang_pair,sample):
            zz = sample[lang_pair]['net_input']['src_tokens']
            uu = self.denoise_tensor(zz, 0.3, 4)
            sample[lang_pair]['net_input']['src_tokens'] = uu
            sample[lang_pair]['net_input']['src_lengths'] = torch.LongTensor([uu.size(1)]*uu.size(0)).cuda()
        
        #^^^^^^^^LSTM^^^^^^^^

        # zz = sample['ar-ar']['net_input']['src_tokens']
        # uu = self.denoise_tensor(zz, 0.2, 3)
        # sample['ar-ar']['net_input']['src_tokens'] = uu
        # sample['ar-ar']['net_input']['src_lengths'] = torch.LongTensor([uu.size(1)]*uu.size(0)).cuda()
        # #^^^^^^^^LSTM^^^^^^^^

        # zz = sample['ke-ke']['net_input']['src_tokens']
        # uu = self.denoise_tensor(zz, 0.2, 3)
        # sample['ke-ke']['net_input']['src_tokens'] = uu
        # sample['ke-ke']['net_input']['src_lengths'] = torch.LongTensor([uu.size(1)]*uu.size(0)).cuda()
        #^^^^^^^^

        # does not work 
        # denoise_input('ar-ar', sample)
        # denoise_input('ke-ke', sample)
        # denoise_input('ar-ke', sample)


        # denoise_input('ke-ar', sample)

        import numpy as np
        
        roll = np.random.random_sample()

        agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}

        for lang_pair in self.args.lang_pairs:
            if sample[lang_pair] is None or len(sample[lang_pair]) == 0:
                continue

            def move_eos_to_beginning (uu):
                ulist = []
                for i in range(uu.size(0)):
                    u = uu[i]
                    u = torch.cat([u[-1:],u[0:-1]], 0)
                    ulist.append(u)

                return torch.stack(ulist).cuda()

            def flip_target(src_pair, sample):

                usample = copy.deepcopy(sample[src_pair])
                u_target = sample[src_pair]['net_input']['src_tokens']
                
                usample['target'] = u_target
                usample['ntokens'] = u_target.size(0) * u_target.size(1)
                usample['net_input']['prev_output_tokens'] = move_eos_to_beginning(u_target)
                # sample[src_pair]['target'] = u_target
                # sample[src_pair]['ntokens'] = u_target.size(0) * u_target.size(1)
                # ssample[src_pair]['net_input']['prev_output_tokens'] = move_eos_to_beginning(u_target)

                return usample

            # if dmodel['nepochs'] < 1:

            if roll < 0.5 and lang_pair == "ar-ke":
                usample = flip_target(lang_pair, sample)
                loss, rewards, sample_size, logging_output = criterion(model.models[lang_pair], usample, lang_pair, dmodel)
                # val = torch.all(sample[lang_pair]['net_input']['src_tokens'].eq(sample[lang_pair]['target']))
            else:
                loss, rewards, sample_size, logging_output = criterion(model.models[lang_pair], sample[lang_pair], lang_pair, dmodel)




            
            # loss, rewards, sample_size, logging_output = criterion(model.models[lang_pair], sample[lang_pair], lang_pair, dmodel)


            if ignore_grad:
                loss *= 0

            optimizer.backward(loss)

            try:
                agg_loss += loss.detach().item()
            except:
                pass

            # TODO make summing of the sample sizes configurable
            agg_sample_size += sample_size
            agg_logging_output[lang_pair] = logging_output
            
        return agg_loss, agg_sample_size, agg_logging_output


    def valid_step(self, sample, model, criterion, dmodel):

        model.eval()

        dev = torch.device("cuda:0")
        
        with torch.no_grad():

            agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}

            for lang_pair in self.eval_lang_pairs:
                if lang_pair not in sample or sample[lang_pair] is None or len(sample[lang_pair]) == 0:
                    continue

                # lang_src = lang_pair.split('-')[0]
                # if lang_pair == 'ar-ke' or lang_pair == 'ke-ke':

                # discriminator, doptimizer, bceloss = dmodel['ke']['model'], dmodel['ke']['optimizer'], dmodel['ke']['loss']
                # loss, rewards, sample_size, logging_output = criterion(model.models[lang_pair], sample[lang_pair], lang_pair, discriminator)
                loss, rewards, sample_size, logging_output = criterion(model.models[lang_pair], sample[lang_pair], lang_pair, dmodel)

                
                agg_loss += loss.data.item()

                # TODO make summing of the sample sizes configurable
                agg_sample_size += sample_size
                agg_logging_output[lang_pair] = logging_output

        return agg_loss, agg_sample_size, agg_logging_output

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        with torch.no_grad():
            return generator.generate(
                    models,
                    sample,
                    prefix_tokens=prefix_tokens,
                    bos_token=_lang_token_index(self.target_dictionary, self.args.target_lang)
                    if self.args.decoder_langtok else self.target_dictionary.eos(),
            )

    def init_logging_output(self, sample):
        return {
            'ntokens': sum(
                sample_lang.get('ntokens', 0)
                for sample_lang in sample.values()
            ) if sample is not None else 0,
            'nsentences': sum(
                sample_lang['target'].size(0) if 'target' in sample_lang else 0
                for sample_lang in sample.values()
            ) if sample is not None else 0,
        }

    def grad_denom(self, sample_sizes, criterion):
        return criterion.__class__.grad_denom(sample_sizes)

    def aggregate_logging_outputs(self, logging_outputs, criterion, logging_output_keys=None):
        logging_output_keys = logging_output_keys or self.eval_lang_pairs
        # aggregate logging outputs for each language pair
        agg_logging_outputs = {
            key: criterion.__class__.aggregate_logging_outputs([
                logging_output.get(key, {}) for logging_output in logging_outputs
            ])
            for key in logging_output_keys
        }

        def sum_over_languages(key):
            return sum(logging_output[key] for logging_output in agg_logging_outputs.values())

        # flatten logging outputs
        flat_logging_output = {
            '{}:{}'.format(lang_pair, k): v
            for lang_pair, agg_logging_output in agg_logging_outputs.items()
            for k, v in agg_logging_output.items()
        }
        flat_logging_output['loss'] = sum_over_languages('loss')
        if any('nll_loss' in logging_output for logging_output in agg_logging_outputs.values()):
            flat_logging_output['nll_loss'] = sum_over_languages('nll_loss')
        flat_logging_output['sample_size'] = sum_over_languages('sample_size')
        flat_logging_output['nsentences'] = sum_over_languages('nsentences')
        flat_logging_output['ntokens'] = sum_over_languages('ntokens')
        return flat_logging_output

    @property
    def source_dictionary(self):
        return self.dicts[self.args.source_lang]

    @property
    def target_dictionary(self):
        return self.dicts[self.args.target_lang]

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        if len(self.datasets.values()) == 0:
            return {'%s-%s' % (self.args.source_lang, self.args.target_lang):
                    (self.args.max_source_positions, self.args.max_target_positions)}
        return OrderedDict([
            (key, (self.args.max_source_positions, self.args.max_target_positions))
            for split in self.datasets.keys()
            for key in self.datasets[split].datasets.keys()
        ])
