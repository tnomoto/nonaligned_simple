# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
from fairseq import utils
from fairseq.data import Dictionary
import torch

from . import FairseqCriterion, register_criterion

import sentencepiece as sp
from collections import Counter 
from nltk.corpus import stopwords
from string import punctuation

# @register_criterion('label_smoothed_cross_entropy_cohere')
@register_criterion('cohere')
class CohereLoss(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.args = args
        
        self.spm = sp.SentencePieceProcessor()
        self.spm.Load("{}/spm.model".format(args.spm_path))
        self.d = Dictionary.load("{}/dict.ar.txt".format(args.dict_path))
        stoplist = stopwords.words('english')
        stoplist += ['-rrb-', '-lrb-', '-lcb-', '-rcb-', '-lsb-','-rsb-', 'mr', 'ms', 'pm','re','am']

        stoplist += [ x.rstrip() for x in open("/opt4/stop-word-list.txt","r") ]
        stoplist = {x:1 for x in set(stoplist) }

        self.stoplist = stoplist


    def spm_decode(self,str):
        out = self.spm.DecodePieces(str.split())
        return out.split()

    def cosine (self, alist, blist):
        import string
        def rm_punct(text):
            text = [x for x in [ x.translate(str.maketrans('','',string.punctuation)) for x in text ] if x ]
            return text 
        a = Counter (rm_punct(alist))
        b = Counter (rm_punct(blist))

        words = list(a.keys() | b.keys())

        a_vec = [a.get(w, 0) for w in words]
        b_vec = [b.get(w, 0) for w in words]

        val_a = sum(v * v for v in a_vec) ** 0.5
        val_b = sum(v * v for v in b_vec) ** 0.5
        dot = sum(a * b for a, b in zip(a_vec, b_vec))

        val_ab = val_a * val_b
        return dot/val_ab if val_ab > 0  else 0.0


    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--dict-path', default=None, type=str, metavar='D')
        parser.add_argument('--spm-path', default=None, type=str, metavar='D')
        
        # fmt: on

    def forward(self, model, sample, reduce=True):

        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        # from sklearn.feature_extraction.text import CountVectorizer
        # from sklearn.feature_extraction import text
        # from sklearn.metrics.pairwise import cosine_similarity


        def token2string(tokenlist):
            sen = []
            for u in tokenlist:
                sen.append(self.d.__getitem__(u))
            return self.spm_decode(' '.join([ w for w in sen if w not in self.stoplist] ))

        n1 = sample['target'].cpu().numpy()
        n2 = sample['net_input']['src_tokens'].cpu().numpy()

        rewards = []
        
        for i in range(n1.shape[0]):
            ar_tok = n2[i].tolist()
            ke_tok = n1[i].tolist()
            val = self.cosine(token2string(ar_tok),token2string(ke_tok))
            rewards.append(val if val > 0.3 else 0.)


        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, rewards, reduce=reduce)

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, rewards, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)

        #***************************

        llprobs = lprobs.clone()
        dim0, dim1, dim2 = lprobs.size()
        llprobs=llprobs.view(lprobs.size(0),-1)
        lprobs=lprobs.view(lprobs.size(0),-1)

        for i in range(llprobs.size(0)):
            llprobs[i] = lprobs[i] + rewards[i]

        lprobs = llprobs
        lprobs = lprobs.view(dim0,dim1,dim2)

        #***************************


        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)

        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss

        return loss, nll_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
