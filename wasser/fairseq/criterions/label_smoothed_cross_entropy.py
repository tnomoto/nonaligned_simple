# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch.nn.functional as F
from fairseq import utils
import torch.nn as nn
import torch
from torch.autograd import Variable
from . import FairseqCriterion, register_criterion



@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.bceloss = nn.BCEWithLogitsLoss(reduction='mean')


    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, lang_pair, dmodel, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        net_output, encoder_out = model(**sample['net_input'])

        cos_loss = None

        if lang_pair == "ar-ke":

            enc_out = encoder_out['encoder_out'][0].transpose(1,0)
            dec_out = net_output[2].transpose(1,0)

            dec_mean = torch.mean(dec_out,1,False)
            enc_mean = torch.mean(enc_out,1,False)

            # cos = nn.CosineSimilarity(dim=1,eps=1e-6)
            # cos = nn.PairwiseDistance(p=2,keepdim=True)

            cos = nn.CosineEmbeddingLoss(reduction='mean')
            cos_loss = cos(dec_mean,enc_mean, torch.Tensor([-1.]).cuda())
            # cos_loss = torch.mean(cos_loss)
            # cos_out = cos(dec_mean,enc_mean)

        rewards = None

        loss, nll_loss = self.label_compute_loss(model, net_output, sample,  reduce=True)

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        

        if cos_loss is not None and dmodel['nepochs']  > 0: 
        # if cos_loss is not None: 
            loss  = (1.0 - cos_loss) + 0.5 * loss

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }

        return loss, rewards, sample_size, logging_output



    def label_compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)


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


    def reinforce_loss(self, model, net_output, sample, rewards, reduce=True):


        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        llprobs = lprobs.clone()
        # print ("rewards", rewards.device.index)
        # print ("llprob", llprobs.device.index)
        # print ("lprob", lprobs.device.index)
        # # import sys
        # sys.exit(0)

        for i in range(lprobs.size(0)):
            for j in range(lprobs.size(1)):

                llprobs[i,j,:] = lprobs[i,j,:] +  (rewards[i,0]) 

        lprobs = llprobs

        # print("llprob {} {}".format(lprobs.size(),model.get_targets(sample, net_output).size()))
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
