# MI-NEE 

Mutual Information Neural Entropic Estimation.
- [Presentation slides](https://ccha23.github.io/MI-NEE/)
- [arXiv paper](https://arxiv.org/abs/1905.12957)

## Dependencies

See [binder/requirements.txt](./binder/requirements.txt)

## Experimental results in Jupyter notebook

- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ccha23/MI-NEE/master?urlpath=lab/tree/MixedGaussian_MINE.ipynb) Mixed Gaussian with MINE
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ccha23/MI-NEE/master?urlpath=lab/tree/MixedGaussian_MINEE.ipynb) Mixed Gaussian with MI-NEE
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ccha23/MI-NEE/master?urlpath=lab/tree/MixedGaussian_MINEE_MINE.ipynb) Mixed Gaussian with cross training by MI-NEE followed by MINE
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ccha23/MI-NEE/master?urlpath=lab/tree/Gaussian_MINE.ipynb) Higher dimensional Gaussian with MINE
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ccha23/MI-NEE/master?urlpath=lab/tree/Gaussian_MINEE.ipynb) Higher dimensional Gaussian with MI-NEE
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ccha23/MI-NEE/master?urlpath=lab/tree/Gaussian_MINEE_MINE.ipynb) Higher dimensional Gaussian with crossing training by MI-NEE followed by MINE
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ccha23/MI-NEE/master?urlpath=lab/tree/other_results.ipynb) Other experimental results 

## Directory structure

- [`data`](./data) folder contains the code that generates samples for the experiments.
  - [`mix_gaussian.py`](./data/mix_gaussian.py) contains the code for the mixed Gaussian distribution.
  - [`gaussian.py`](./data/gaussian.py) contains the code for gaussian distribution.
- [`model`](./model) folder contains the code of the mutual information estimators.
  - [`mine.py`](./model/mine.py) contains the code for MINE.
  - [`minee.py`](./model/minee.py) contains the code for MI-NEE.
  - [`minee_mine.py`](./model/minee_mine.py) contains the code for cross training by MI-NEE followed by MINE.

## Reference

Chung Chan, Ali Al-Bashabsheh, Hing Pang Huang, Michael Lim, Da Sun Handason Tam, and Chao Zhao. "Neural Entropic Estimation: A faster path to mutual information estimation." arXiv preprint [arXiv:1905.12957](https://arxiv.org/abs/1905.12957) (2019).