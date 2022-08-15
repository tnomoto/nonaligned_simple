#!/bin/bash
export PYTHONPATH=./fairseq_base_monte_carlo/fairseq:$PYTHONPATH
VERSION=0
POSITIONAL=()
TURK=
while [[ $# -gt 0 ]]
do
    key=$1
    case $key in

	-d|--dir)
	    DIR="$2"
	    shift # past argument
	    shift # past value
	    ;;

	-n|--num_hypo)
	    N_HYPO="$2"
	    shift # past argument
	    shift # past value
	    ;;

	-m|--model)
	    MODEL="$2"
	    shift # past argument
	    shift # past value
	    ;;
	-o|--use_turk)
	    TURK=1
	    shift # past argument
	    shift # past value
	    ;;

	-v|--version)
	    VER="$2"
	    shift # past argument
	    shift # past value
	    ;;

	*)    # unknown option
	    POSITIONAL+=("$1") # save it in an array for later
	    shift # past argument
	    echo "-d <data> -m <model> [-o]"
	    exit
	    ;;
    esac
done

set -- "${POSITIONAL[@]}" # restore positional parameters
	

PWD=`pwd`
DX=${PWD}/data/${DIR}
SPM=${PWD}/data/${DIR}/xmlcnn/data/spm.model
echo "using $SPM"
jconv=${MODEL}
mode=modeless

# echo "TURK $TURK"
# exit
# (cd fairseq_20190530
# ~/nn_generate.sh -d ${DIR} -m ${MODEL} -g 1
#cd # )


# ${PWD}/myinteractive.sh  -v ${VER}

${PWD}/myinteractive.sh -m ${MODEL} -v ${VER} -o ${DX}/nn_pred_${MODEL}_modeless.out 


echo D=${DX} MODEL=${MODEL} MODE=${mode}

if test -z "$TURK" ; then

	~/bleu/spm_write_riper.py -o $DX/riper_wiki.txt -r $DX/xmlcnn/data/n_test.ar -a $DX/nn_pred_${jconv}_modeless.out  -n ${N_HYPO} -p ${SPM} 
else

	~/bleu/spm_write_riper.py -o $DX/riper_wiki.txt -r $DX/xmlcnn/data/turk_test.ar -a $DX/nn_pred_${jconv}_modeless.out  -n 1 -p ${SPM} 
fi	


export JAVA_HOME=/usr/lib/jvm/default-java
export JOSHUA=~/ppdb-simplification-release-joshua5.0/joshua
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

if [ ! -z "$TURK" ]; then
	~/ppdb-simplification-release-joshua5.0/joshua/bin/bleu $DX/riper_wiki.txt ~/simplification/data/turkcorpus/test.8turkers.tok.turk 8
#~/wikilarge_test/wiki_sari.py -s $DX/riper_wiki.txt -t ~/wikilarge_test/turkcorpus
# export JAVA_HOME=/usr/lib/jvm/default-java
# export JOSHUA=~/ppdb-simplification-release-joshua5.0/joshua
# export LC_ALL=en_US.UTF-8
# export LANG=en_US.UTF-8
	sysdir=${DX}/sari
	input=~/simplification/data/turkcorpus/test.8turkers.tok.norm
	ref=~/simplification/data/turkcorpus/test.8turkers.tok.turk

	(cd ~/ppdb-simplification-release-joshua5.0/joshua/bin
 	./star $DX/riper_wiki.txt  $ref $input | grep STAR
	)

	~/bleu/flesch_reading_ease.py -f $DX/riper_wiki.txt
	~/bleu/readability_test.py <  $DX/riper_wiki.txt

else
	~/bleu/nltk_simpl_scores.py -o ${DX}/riper_wiki.txt  \
	-t ${DX}/xmlcnn/data/n_test.ke \
	-s ${DX}/xmlcnn/data/n_test.ar


	input=${DX}/xmlcnn/data/n_test.ar
	ref=${DX}/xmlcnn/data/n_test.ke

	(cd ~/ppdb-simplification-release-joshua5.0/joshua/bin
 	./star_1 $DX/riper_wiki.txt $ref $input |grep STAR
	)

fi
