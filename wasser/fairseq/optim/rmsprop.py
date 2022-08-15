# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch.optim

from . import FairseqOptimizer, register_optimizer


@register_optimizer('rmsprop')
class RMSPROP(FairseqOptimizer):
    def __init__(self, args, params):
        super().__init__(args, params)
        self._optimizer = torch.optim.RMSprop(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        parser.add_argument('--momentum', default=0.0, type=float, metavar='M',
                            help='momentum factor')
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
        parser.add_argument('--rms-centered', action='store_true',default=False,
                            help='centered switch')
        parser.add_argument('--rms-eps', default=1e-08, type=float, metavar='E',
                            help='eps factor')
        parser.add_argument('--rms-alpha', default=0.99, type=float, metavar='A',
                            help='alpha factor')

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            'lr': self.args.lr[0],
            'momentum': self.args.momentum,
            'weight_decay': self.args.weight_decay,
            'centered':self.args.rms_centered,
            'eps':self.args.rms_eps,
            'alpha':self.args.rms_alpha,

        }
