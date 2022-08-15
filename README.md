# Simplification with Non-Aligned Data

This is a source code associated with a paper with a title: <br/>
[**Learning to Simplify with Data Hopelessly Out of Alignment**](https://arxiv.org/2204.00741).  

## Pre-requisites

* Python 3.6
* torch                   1.3.1
* torchaudio              0.10.0+cu113
* torchtext               0.5.0
* torchvision             0.11.1+cu113



## How to run 


### Generation

* js-gan

```bash
cd js
../util/generate.sh -t tsd

```

* wasser-gan

```bash
cd wasser
../util/generate.sh -t tsd

````




