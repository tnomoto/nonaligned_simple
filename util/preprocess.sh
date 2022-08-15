#!/bin/bash

POSITIONAL=()
while [[ $# -gt 0 ]]
do
    key=$1
    case $key in
	-d|--datadir)
	    data_dir="$2"
	    shift 
	    shift 
	    ;;
	*)    # unknown option
	    POSITIONAL+=("$1") 
	    echo "-d <data_dir> -g <gpu>"
	    shift 
	    ;;
    esac
done

set -- "${POSITIONAL[@]}" # restore positional parameters


pair=ar_ke

DATA_D=`pwd`/data
SRCDIR=${DATA_D}/${data_dir}/xmlcnn/data
DEST=${DATA_D}/${data_dir}/data_bin


python preprocess.py --source-lang ar --target-lang ke \
    --trainpref $SRCDIR/train.ar_ke --validpref $SRCDIR/valid.ar_ke \
    --joined-dictionary \
    --destdir ${DEST}\
    --workers 4



python preprocess.py --source-lang ke --target-lang ke \
    --trainpref $SRCDIR/train.ke_ke --validpref $SRCDIR/valid.ke_ke \
    --tgtdict ${DEST}/dict.ke.txt\
    --srcdict ${DEST}/dict.ke.txt\
    --destdir ${DEST}\
    --workers 4

