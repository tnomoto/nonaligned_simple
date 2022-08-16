#!/bin/bash


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

	*)    # unknown option
	    POSITIONAL+=("$1") 
	    shift 
	    echo "-d <data> -m <model> [-o]"
	    exit
	    ;;
    esac
done

set -- "${POSITIONAL[@]}" # restore positional parameters
	

PWD=`pwd`
DX=${PWD}/data/${DIR}
SPM=${PWD}/data/${DIR}/xmlcnn/data/spm.model


../util/spm_write_riper.py -o $DX/pred_decoded.txt -r $DX/xmlcnn/data/oo_test.ar -a $DX/pred.out  -n 1 -p ${SPM} 



