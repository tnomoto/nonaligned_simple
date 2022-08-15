# Simplification with Non-Aligned Data

This is a source code associated with a paper with a title: <br/>
[**Learning to Simplify with Data Hopelessly Out of Alignment**](learn.html).  

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
../util/generate.sh -d tsd

```

* wasser-gan

```bash
cd wasser
../util/generate.sh -d tsd

````

### Training


```bash
cd js
../util/train.sh -d tsd -b 64 -g 1

```

* wasser-gan

```bash
cd wasser
../util/train.sh -d tsd -b 64 -g 1

````


