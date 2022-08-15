#!/bin/bash


POSITIONAL=()
while getopts "v:m:o:d:" o; do
    case "${o}" in
	# v)
	#     VER=${OPTARG}
	#     ;;

	d)
	    DD=${OPTARG}
	    ;;

	# m)
	#     MODEL=${OPTARG}
	#     ;;
	# o)
	#     OUTPUT=${OPTARG}
	#     ;;
	*)
	    usage
	    ;;
    esac
done



PRJ=data/${DD}
BIN=.
#PRJ=/opt3/jamai
#PRJ=/opt3/unsup_wikilarge_voc8000
#PRJ=/opt3/wikilarge_voc1000

#PRJ=`pwd`/data/tsd_300k
#PRJ=`pwd`/data/tsd_200k
# PRJ=`pwd`/data/tsd_100k
#PRJ=`pwd`/data/yelp
MODEL=multilingual_lstm
VER=_best
# echo "using checkpoints ${VER}"
# echo "using checkpoints ${MODEL}"
# echo "using checkpoints ${OUTPUT}"

echo $PRJ


CUDA_VISIBLE_DEVICES=1 python ${BIN}/interactive.py ${PRJ}/data_bin \
      --task multilingual_translation --source-lang ar --target-lang ke\
      --path ${PRJ}/checkpoints/${MODEL}/checkpoint${VER}.pt \
      --batch-size 1 \
      --lang-pairs ar-ke \
      --nbest 1 \
      --beam 5  < ${PRJ}/xmlcnn/data/test.ar 
