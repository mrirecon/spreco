# Sampling Posterior for MRI Reconstruction (SPRECO)

This package is to help you extract prior information for MRI image and then use it for MRI reconstruction. 

# Installation

1. Clone this repository and use [conda](https://www.anaconda.com/products/individual) to set up the environment.

   ```shell
   $ git clone https://github.com/mrirecon/spreco.git
   $ cd spreco
   $ conda env create --file env.yml -n work
   $ conda activate work
   $ pip install .
   ```

2. Install with pip command
   ```shell
   $ pip install spreco
   ```

## Quickstart with colab

1. Sample the posterior [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrirecon/spreco/blob/main/scripts/demo_recon.ipynb)
2. Train an image prior [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrirecon/spreco/blob/main/scripts/demo_train.ipynb)

## Reference 
Please use tag "mrm_paper" to get code to generate results in following references. We would appreciate it if you used our codes and cited our work.

[1] Luo, G, Blumenthal, M, Heide, M, Uecker, M. Bayesian MRI reconstruction with joint uncertainty estimation using diffusion models. Magn Reson Med. 2023; 1-17

[2] Luo, G, Zhao, N, Jiang, W, Hui, ES, Cao, P. MRI reconstruction using deep Bayesian estimation. Magn Reson Med. 2020; 84: 2246–2261.

