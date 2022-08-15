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

