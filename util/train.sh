#!/bin/bash

usage() { echo "Usage: $0  [-d <data> -m <model> -b <batch> -g <gpu> -x <anneal_point>]" ; exit 1; }
export PYTHONPATH=~/bleu/:$PYTHONPATH

while getopts ":d:b:g:" o; do
    case "${o}" in
	d)
	    train_dir_0=${OPTARG}
	    ;;
	b)
	    batch_size=${OPTARG}
	     ;;
	g)
	    GPUDEV=${OPTARG}
	     ;;
	*)
	    usage
	    ;;
    esac
done


data_dir=./resource/${train_dir_0}

#batch_size=128

shift $((OPTIND-1))

if [ -z "${train_dir_0}" ] ; then
    usage
fi

MODEL=multilingual_lstm


CUDA_VISIBLE_DEVICES=${GPUDEV} python ./train.py ${data_dir}/data_bin/ \
		    --max-epoch 12\
		    --ddp-backend=no_c10d \
		    --task multilingual_translation --lang-pairs ar-ke,ke-ke\
		    --arch ${MODEL} \
		    --share-decoder-input-output-embed \
		    --optimizer adam --adam-betas '(0.9, 0.98)' \
		    --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
		    --warmup-updates 4000 --warmup-init-lr '1e-07' \
		    --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
		    --dropout 0.3 --weight-decay 0.0001 \
		    --save-dir ${data_dir}/checkpoints/${MODEL}\
		    --max-tokens 1000 \
		    --decoder-embed-dim 512 \
		    --encoder-embed-dim 512 \
		    --share-all-embeddings \
		    --share-decoders \
		    --share-encoders \
		    --batch-size ${batch_size} \
	        --update-freq 3\
	        --d-path ${data_dir}/checkpoints/discriminator


