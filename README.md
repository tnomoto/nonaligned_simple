# Simplification with Non-Aligned Data

This is a source code that supplements a paper: <br/>
[**Learning to Simplify with Data Hopelessly Out of Alignment**](https://arxiv.org/2204.00741), published on ArXiv. 

## Pre-requisites

* Python 3.6
* torch                   1.3.1
* torchaudio              0.10.0+cu113
* torchtext               0.5.0
* torchvision             0.11.1+cu113
* sentencepiece           0.1.9 (together with its python library)


## Setting it up 

Download and untar the following at the project's top directory.

* [asset.tar.bz2](https://1drv.ms/u/s!Ahv4QmK34dXhgRH_zzprmcuByqm4)
* [js-resource.tar.bz2](https://1drv.ms/u/s!Ahv4QmK34dXhgRIUDqy4Q39JAXIg)
* [wasser-resource.tar.bz2](https://1drv.ms/u/s!Ahv4QmK34dXhgRSuiAp--PeQbDUr)

```bash
cd nonaligned_simple
tar jxvf asset.tar.bz2
tar jxvf js-resource.tar.bz2
tar jxvf wasser-resource.tar.bz2

```


## Training


```bash
cd js
../util/train.sh -d tsd -b 64 -g 1

```

* wasser-gan

```bash
cd wasser
../util/train.sh -d tsd -b 64 -g 1

````

-d : dataset name <br/>
-b : batch size<br/>
-g : GPU ID<br/>

## Generation

### Using pre-trained models


As a trained model is provided as part of the package, you can bypass training and go to the generation step directly. Here is what you do. 

* js-gan

```bash
cd js
../util/generate.sh -d tsd -g 1

```
The output is found in js/data/tsd/pred.out.

* wasser-gan

```bash
cd wasser
../util/generate.sh -d tsd -g 1

````
-d : dataset name <br/>
-g : GPU ID <br/>

The result in wasser/data/tsd/pred.out.

### Detokenization

***generate.sh*** gives you a result in a sentence-piece format. Running the following will restore it into a normal text.

```bash
cd js
../util/decode_spm.sh 
````
You find the result in js/data/tsd/pred_decoded.txt.

## About Data

The training data is based on [the sscorpus](https://github.com/tmu-nlp/sscorpus). We removed source/target pairs whose similarity exceeds 0.65. The test set is the same as one used by Zhang, et al (2017). 


## References

```bibtex
@misc{https://doi.org/10.48550/arxiv.2204.00741,
doi = {10.48550/ARXIV.2204.00741},
url = {https://arxiv.org/abs/2204.00741},
author = {Nomoto, Tadashi},
keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
title = {Learning to Simplify with Data Hopelessly Out of Alignment},
publisher = {arXiv},
year = {2022},
copyright = {Creative Commons Attribution Share Alike 4.0 International}
}
```
```bibtex
@inproceedings{zhang-lapata-2017-sentence,
title = "Sentence Simplification with Deep Reinforcement Learning",
author = "Zhang, Xingxing  and Lapata, Mirella",
booktitle = "Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing",
month = sep,
year = "2017",
address = "Copenhagen, Denmark",
publisher = "Association for Computational Linguistics",
url = "https://aclanthology.org/D17-1062",
 }
 ```
