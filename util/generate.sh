#!/bin/bash


POSITIONAL=()
while getopts "d:g:" o; do
    case "${o}" in
	# v)
	#     VER=${OPTARG}
	#     ;;

	d)
	    DD=${OPTARG}
	    ;;

	g)
	    GPU=${OPTARG}
	    ;;
	# o)
	#     OUTPUT=${OPTARG}
	#     ;;
	*)
	    usage
	    ;;
    esac
done



PRJ=./resource/${DD}
BIN=.

MODEL=multilingual_lstm
VER=_best


echo $PRJ


CUDA_VISIBLE_DEVICES=${GPU} python ${BIN}/interactive.py ${PRJ}/data_bin \
      --task multilingual_translation --source-lang ar --target-lang ke\
      --path ${PRJ}/checkpoints/${MODEL}/checkpoint${VER}.pt \
      --batch-size 1 \
      --lang-pairs ar-ke \
      --nbest 1 \
      --beam 5  < ${PRJ}/xmlcnn/data/test.ar 
