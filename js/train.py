#!/usr/bin/env python
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Train a new model on one or across multiple GPUs.
"""

import collections
import math
import os
import random

# import adamod 
from time import sleep
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter
from fairseq.models.simple_cnn import SimpleCNN
from fairseq.models.simple_cnn import SimpleCNN_W
from fairseq.models.simple_cnn import SimpleLinear




def main(args, init_distributed=False):
    utils.import_user_module(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)
    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)

    # Print args
    # print(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)

    # dev = torch.device("cuda:0")

    # Single GPU model


    discriminator_k = SimpleCNN(args.encoder_embed_dim).cuda()

    # discriminator_k = SimpleCNN(len(task.dicts['ar'])).cuda()
    # doptimizer_k = torch.optim.Adam(discriminator_k.parameters(),lr=0.0002)

    params = list(discriminator_k.parameters())
    doptimizer_k = torch.optim.Adam(params,lr=0.0002)

    # doptimizer_k = torch.optim.RMPS(params,lr=0.0002)
    bceloss_k = torch.nn.BCELoss(reduction='mean')



    if os.path.isfile("{}/{}".format(args.d_path,'checkpoint_k.pt')):

        dstate = torch.load("{}/{}".format(args.d_path,'checkpoint_k.pt'))
        discriminator_k.load_state_dict(dstate['model'])
        doptimizer_k.load_state_dict(dstate['optimizer'])
        print("reloading discriminator")



    dmodel_k ={'model': discriminator_k, 'loss': bceloss_k, 'optimizer': doptimizer_k, 'params':params}



    discriminator_a = SimpleCNN_W(args.encoder_embed_dim).cuda()
    discriminator_a.to(torch.device("cuda:0"))
    params = list(discriminator_a.parameters())

    doptimizer_a = torch.optim.RMSprop(params,lr=0.00005)



    bceloss_a = torch.nn.BCELoss(reduction='mean')

    if os.path.isfile("{}/{}".format(args.d_path,'checkpoint_a.pt')):

        dstate = torch.load("{}/{}".format(args.d_path,'checkpoint_a.pt'))
        discriminator_a.load_state_dict(dstate['model'])
        doptimizer_a.load_state_dict(dstate['optimizer'])
        print("reloading discriminator")

    dmodel_a ={'model': discriminator_a, 'loss': bceloss_a, 'optimizer': doptimizer_a, "params":params}




    linear_c = SimpleLinear(args.encoder_embed_dim).cuda()
    params = list(linear_c.parameters())
    optim = torch.optim.Adam(params,lr=0.0002)
    bceloss = torch.nn.BCELoss(reduction='mean')


    dmodel_c ={'model':linear_c, 'optimizer': optim, "loss":bceloss}
    dmodel = {"ar":dmodel_a,"ke":dmodel_k,'model_c':dmodel_c}

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=True, epoch=0)

 
    # print(model)

    # print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    # print('| num. model params: {} (num. trained: {})'.format(
    #     sum(p.numel() for p in model.parameters()),
    #     sum(p.numel() for p in model.parameters() if p.requires_grad),
    # ))

    # Build trainer
    trainer = Trainer(args, task, model, criterion)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_losses = [None]
    valid_subsets = args.valid_subset.split(',')

    nsteps = 0 

    while lr > args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update:
    # while lr > args.min_lr  and trainer.get_num_updates() < max_update:
       
        # train for one epoch

        # nsteps += 1
        # print(nsteps)
        # continue

        train(args, epoch_itr.epoch, trainer, task, epoch_itr, dmodel, model, criterion, nsteps)
         # model.models[src_pair].encoder.eval() += 1
     
        if not args.disable_validation and epoch_itr.epoch % args.validate_interval == 0:

            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets, dmodel)

        else:
            valid_losses = [None]

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:

            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])


            dsave_a = {'model':dmodel['ar']['model'].state_dict(),
                     'optimizer':dmodel['ar']['optimizer'].state_dict(), 
                     'loss':dmodel['ar']['loss']}


            dsave_k = {'model':dmodel['ke']['model'].state_dict(),
                     'optimizer':dmodel['ke']['optimizer'].state_dict(), 
                     'loss':dmodel['ke']['loss']}
# 
            torch.save(dsave_a, "{}/{}".format(args.d_path,'checkpoint_a.pt'))
            torch.save(dsave_k, "{}/{}".format(args.d_path,'checkpoint_k.pt'))


        if ':' in getattr(args, 'data', ''):
            # sharded data: get train iterator for next epoch
            epoch_itr = trainer.get_train_iterator(epoch_itr.epoch)

    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))


def train(args, epch, trainer, task, epoch_itr, dmodel, model, criterion, nsteps):
    """Train the model for one epoch."""
    # Update parameters every N batches



    update_freq = args.update_freq[epoch_itr.epoch - 1] \
        if epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1]

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.epoch >= args.curriculum),
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    valid_subsets = args.valid_subset.split(',')
    max_update = args.max_update or math.inf

    generator = task.build_generator(args)

    # fout = open("discriminator.out","w")
    
    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):

        log_output, d_states  = trainer.train_step(epch, i, generator, samples, dmodel)

        # dev = torch.device("cuda:0")

        for i, sample in enumerate(samples):

            if sample is None:
                continue

            sample = utils.move_to_cuda(sample)


            ## JS Gan
            
            for src_pair in args.lang_pairs:

                if src_pair == 'ke-ke' or  src_pair == 'ar-ke':

                    discriminator, doptimizer, bceloss = dmodel['ke']['model'], dmodel['ke']['optimizer'], dmodel['ke']['loss']
                    linear_c = dmodel['model_c']['model']

                    discriminator.train()
                    doptimizer.zero_grad()

                    for p in model.models[src_pair].encoder.parameters():
                        p.data.requires_grad = False
                   

                    src_tokens = sample[src_pair]['net_input']['src_tokens']
                    src_lengths = sample[src_pair]['net_input']['src_lengths']
                    encoder_out = model.models[src_pair].encoder(src_tokens, src_lengths)


                    # prev_output_tokens = sample[src_pair]['net_input']['prev_output_tokens']
                    # decoder_out = model.models[src_pair].decoder(prev_output_tokens,encoder_out=encoder_out)
                    # seq_out = decoder_out[1]['inner_states'][-1].clone().detach()
                    # seq_out = encoder_out['encoder_out'].clone().detach()

                    seq_out = encoder_out['encoder_out'][1]#T x B x C
                    seq_out = seq_out.clone().detach()
                    seq_out = seq_out.permute(1,0,2)

                    d_lim = 1000 if seq_out.size(1) > 1000 else seq_out.size(1)
                    cnn_input = torch.zeros(seq_out.size(0),1000,seq_out.size(2))
                    cnn_input[:,:d_lim, :] = seq_out[:,:d_lim,:]

                    hypo = cnn_input.cuda()

                    rewards = discriminator(hypo)  
                    # fout.write("SRC1 {}\n".format(src_pair))
                    # fout.write("{}\n".format(rewards) )
                    target =  torch.zeros(rewards.size()) if src_pair == 'ar-ke' else torch.ones(rewards.size())
                    dloss = bceloss(rewards, target.cuda())

                    dloss.backward()
                    doptimizer.step()


                if  src_pair == 'ar-ke' :

                    discriminator, doptimizer, bceloss = dmodel['ke']['model'], dmodel['ke']['optimizer'], dmodel['ke']['loss']
                    discriminator.eval()
                    doptimizer.zero_grad()
                    model.models[src_pair].encoder.train()

                    # model.models[src_pair].encoder.train()
                    model.models[src_pair].encoder.zero_grad()

                    src_tokens = sample[src_pair]['net_input']['src_tokens']
                    src_lengths = sample[src_pair]['net_input']['src_lengths']
                    encoder_out = model.models[src_pair].encoder(src_tokens, src_lengths)
                    # prev_output_tokens = sample[src_pair]['net_input']['prev_output_tokens']
                    # decoder_out = model.models[src_pair].decoder(prev_output_tokens,encoder_out=encoder_out)
                    # seq_out = decoder_out[1]['inner_states'][-1]
                    # seq_out = encoder_out['extra']
                    seq_out = encoder_out['encoder_out'][1] #T x B x C

                    seq_out = seq_out.permute(1,0,2)

                    d_lim = 1000 if seq_out.size(1) > 1000 else seq_out.size(1)
                    cnn_input = torch.zeros(seq_out.size(0),1000,seq_out.size(2))
                    cnn_input[:,:d_lim, :] = seq_out[:,:d_lim,:]
                    hypo = cnn_input.cuda()
                    # linear_c, doptimizer, bceloss = dmodel['model_c']['model'], dmodel['model_c']['optimizer'], dmodel['model_c']['loss']
                    # hypo = linear_c(hypo)
                    rewards = discriminator(hypo)  
                    target =  torch.ones(rewards.size()).cuda()

                    # dloss =  (0.1 * nsteps/ 100000) * bceloss(rewards, target)
                    dloss =  bceloss(rewards, target)


                    dloss.backward()
                    trainer.optimizer.step()


                model.models[src_pair].encoder.train()

            #     ### +++++++++++++++++++++++++++++++++++++++++++++++++++++++

            

                # if  src_pair == 'ar-ke' or src_pair == 'ar-ar':
                #     continue

                #     # discriminator, doptimizer, bceloss = dmodel['ke']['model'], dmodel['ke']['optimizer'], dmodel['ke']['loss']
                #     discriminator, doptimizer, bceloss = dmodel['ar']['model'], dmodel['ar']['optimizer'], dmodel['ar']['loss']

                #     doptimizer.zero_grad()

                #     src_tokens = sample[src_pair]['net_input']['src_tokens']
                #     src_lengths = sample[src_pair]['net_input']['src_lengths']
                #     encoder_out = model.models[src_pair].encoder(src_tokens, src_lengths)
                #     prev_output_tokens = sample[src_pair]['net_input']['prev_output_tokens']
                #     decoder_out = model.models[src_pair].decoder(prev_output_tokens,encoder_out=encoder_out)
                    
                #     # seq_out = decoder_out[1]['inner_states'][-1].clone().detach()
                #     # seq_out = encoder_out['encoder_out'].clone().detach()
                #     # print(">>>", type(decoder_out))
                #     # import sys
                #     # sys.exit(0)

                #     seq_out = decoder_out[2]


                #     # seq_out = decoder_out[1]['inner_states'][0]
                #     # for i in range(1,len(decoder_out[1]['inner_states'])):
                #     #     seq_out.add_(decoder_out[1]['inner_states'][i])

                #     seq_out = seq_out.permute(1,0,2)

                #     d_lim = 1000 if seq_out.size(1) > 1000 else seq_out.size(1)
                #     cnn_input = torch.zeros(seq_out.size(0),1000,seq_out.size(2))
                #     cnn_input[:,:d_lim, :] = seq_out[:,:d_lim,:]
                #     hypo = Variable(cnn_input, requires_grad=True).cuda()

                #     rewards = discriminator(hypo)  
                #     dloss =  torch.mean(rewards)

                #     if src_pair == "ar-ke":
                #         dloss *= -1

                #     fout.write("SRC2-1 {}\n".format(src_pair))
                #     fout.write("{}\n".format(rewards))
                #     dloss.backward()
                #     torch.nn.utils.clip_grad_value_(dmodel['ar']['params'], 0.01)
                #     doptimizer.step()

                    
                    # target =  torch.zeros(rewards.size()) if src_pair == 'ar-ke' else torch.ones(rewards.size())
                    # dloss = bceloss(rewards, target.to(dev))

            #     # if  src_pair == 'ar-ke':
            #     #     discriminator, doptimizer, bceloss = dmodel['ar']['model'], dmodel['ar']['optimizer'], dmodel['ar']['loss']
            #     #     doptimizer.zero_grad()
            #     #     src_tokens = sample[src_pair]['net_input']['src_tokens']
            #     #     src_lengths = sample[src_pair]['net_input']['src_lengths']
            #     #     encoder_out = model.models[src_pair].encoder(src_tokens, src_lengths)
            #     #     prev_output_tokens = sample[src_pair]['net_input']['prev_output_tokens']
            #     #     decoder_out = model.models[src_pair].decoder(prev_output_tokens,encoder_out=encoder_out)
                    
            #     #     seq_out = decoder_out[1]['inner_states'][0]

            #     #     for i in range(1,len(decoder_out[1]['inner_states'])):
            #     #         seq_out.add_(decoder_out[1]['inner_states'][i])


            #     #     seq_out = seq_out.permute(1,0,2)

            #     #     d_lim = 1000 if seq_out.size(1) > 1000 else seq_out.size(1)
            #     #     cnn_input = torch.zeros(seq_out.size(0),1000,seq_out.size(2))
            #     #     cnn_input[:,:d_lim, :] = seq_out[:,:d_lim,:]
            #     #     hypo = Variable(cnn_input, requires_grad=True).cuda()

            #     #     rewards = discriminator(hypo)  
            #     #     fout.write(">>SRC2-2 {}\n".format(src_pair))
            #     #     fout.write("{}\n".format(rewards) )

            #     #     target =  torch.ones(rewards.size())
            #     #     # dloss = bceloss(1 - rewards, target.to(dev))
            #     #     # dloss = -(0.001 * nsteps/50000) * bceloss(rewards, target.to(dev))
            #     #     dloss = bceloss(rewards, target.to(dev))

            #     #     # dloss = -bceloss(rewards, target.to(dev))

            #     #     dloss.backward()
            #     #     doptimizer.step()


        if log_output is None:
            continue

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                continue  # these are already logged above
            if 'loss' in k:
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats, tag='train', step=stats['num_updates'])

        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()

        num_updates = trainer.get_num_updates()
        
        if (
            not args.disable_validation
            and args.save_interval_updates > 0
            and num_updates % args.save_interval_updates == 0
            and num_updates > 0
        ):
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets, dmodel)
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break


    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats, tag='train', step=stats['num_updates'])

    # reset training meters
    for k in [
        'train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip',
    ]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()


def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('train_loss')
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('train_loss')
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['wps'] = trainer.get_meter('wps')
    stats['ups'] = trainer.get_meter('ups')
    stats['wpb'] = trainer.get_meter('wpb')
    stats['bsz'] = trainer.get_meter('bsz')
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = trainer.get_meter('gnorm')
    stats['clip'] = trainer.get_meter('clip')
    stats['oom'] = trainer.get_meter('oom')
    if trainer.get_meter('loss_scale') is not None:
        stats['loss_scale'] = trainer.get_meter('loss_scale')
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    stats['train_wall'] = trainer.get_meter('train_wall')
    return stats


def validate(args, trainer, task, epoch_itr, subsets, dmodel):
    """Evaluate the model on the validation set(s) and return the losses."""
    valid_losses = []
    for subset in subsets:
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        # reset validation loss meters
        for k in ['valid_loss', 'valid_nll_loss']:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        extra_meters = collections.defaultdict(lambda: AverageMeter())


        for sample in progress:


            log_output = trainer.valid_step(sample, dmodel)

            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                    continue
                extra_meters[k].update(v)

        # log validation stats
        stats = get_valid_stats(trainer)
        for k, meter in extra_meters.items():
            stats[k] = meter.avg
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats['loss'].avg)
    return valid_losses


def get_valid_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('valid_loss')
    if trainer.get_meter('valid_nll_loss').count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = stats['loss']
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, 'best'):
        stats['best_loss'] = min(
            checkpoint_utils.save_checkpoint.best, stats['loss'].avg)
    return stats


def distributed_main(i, args, start_rank=0):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)


def cli_main():
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)

    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        assert args.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
            print('| NOTE: you may get better performance with: --ddp-backend=no_c10d')
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args, ),
            nprocs=args.distributed_world_size,
        )
    else:
        # single GPU training
        main(args)


if __name__ == '__main__':
    cli_main()
