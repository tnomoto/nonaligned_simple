#!/bin/bash

VERSION=0
POSITIONAL=()

while [[ $# -gt 0 ]]
do
    key=$1
    case $key in

	-v|--voc) # eg. 8000
	    VOC="$2"
	    shift # 
	    shift # 
	    ;;
	-p|--prefix)
	    PREFIX="$2"
	    shift #
	    shift # 
	    ;;

	*)    # unknown option
	    POSITIONAL+=("$1") # save it in an array for later
	    shift # past argument
	    echo "-v <vocab_size>"
	    exit
	    ;;
    esac
done

set -- "${POSITIONAL[@]}" # restore positional parameters

head -250000 ss_redacted.csv | cut -d@ -f1 > ${PREFIX}_train.ar 
head -250000 ss_redacted.csv | cut -d@ -f2 > ${PREFIX}_train.ke 

tail -1749 ss_redacted.csv | cut -d@ -f1> ${PREFIX}_valid.ar
tail -1749 ss_redacted.csv | cut -d@ -f2> ${PREFIX}_valid.ke


cat ${PREFIX}*.{ar,ke} > full_corpus.txt

spm_train --input=full_corpus.txt --model_prefix=spm --vocab_size=${VOC} --character_coverage=1.0

for f in ${PREFIX}*.{ar,ke}; do vif=${f#*_};spm_encode --model=spm.model --output_format=piece < $f > $vif; done 
