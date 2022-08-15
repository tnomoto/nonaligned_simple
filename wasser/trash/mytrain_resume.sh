#!/bin/bash

usage() { echo "Usage: $0  [-d <data> -m <model> -b <batch> -g <gpu> -x <anneal_point>]" ; exit 1; }
export PYTHONPATH=~/bleu/:$PYTHONPATH

while getopts ":d:b:" o; do
    case "${o}" in
	d)
	    train_dir_0=${OPTARG}
	    ;;
	b)
	    batch_size=${OPTARG}
	     ;;
	*)
	    usage
	    ;;
    esac
done

# data_dir=/opt1/${train_dir_0}
data_dir=/opt3/${train_dir_0}


#batch_size=128

shift $((OPTIND-1))

#if [ -z "${GPUDEV}" ] ||[ -z "${train_dir_0}" ] ; then
if [ -z "${train_dir_0}" ] ; then
    usage
fi

MODEL=multilingual_fconv_pg 

# if [ -e "${data_dir}/checkpoints/${MODEL}" ];then
#     rm -r ${data_dir}/checkpoints/${MODEL}
# fi


# if [ -e "${data_dir}/checkpoints/discriminator" ];then
# 	rm -r ${data_dir}/checkpoints/discriminator
# fi

# mkdir -p ${data_dir}/checkpoints/discriminator

CUDA_VISIBLE_DEVICES=0,1,2 python ./train.py ${data_dir}/data_bin/ \
		    --max-epoch 30\
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
		    --max-tokens 800 \
		    --share-encoders \
		    --share-all-embeddings \
		    --batch-size ${batch_size} \
	        --update-freq 3\
	        --d-path ${data_dir}/checkpoints/discriminator


