# Generative Pre-trained Image Prior for MRI reconstruction

This package is to help you extract prior information for MRI image and then use it for MRI reconstruction. 

# Installation

1. Install with pip command
   ```shell
   $ pip install spreco
   ```

2. Clone this repository and use [conda](https://www.anaconda.com/products/individual) to set up the environment.

   ```shell
   $ git clone https://github.com/mrirecon/spreco.git
   $ cd spreco
   $ conda env create --file env.yml -n work
   $ conda activate work
   $ pip install .
   ```

## Quickstart with colab

1. Sample the posterior 
   - [Jupyter Notebook](https://github.com/mrirecon/spreco/blob/main/examples/scripts/demo_recon.ipynb)
   - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrirecon/spreco/blob/main/examples/scripts/demo_recon.ipynb)
2. Train an image prior
   - [Jupyter Notebook](https://github.com/mrirecon/spreco/blob/main/examples/scripts/demo_train.ipynb)
   - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrirecon/spreco/blob/main/examples/scripts/demo_train.ipynb)
3. Using Prior with BART
   - [Jupyter Notebook](https://github.com/mrirecon/bart-workshop/blob/master/ismrm2021/bart_tensorflow/bart_tf.ipynb)
   - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrirecon/bart-workshop/blob/master/ismrm2021/bart_tensorflow/bart_tf.ipynb)
## Reference 
We would appreciate it if you tried our codes and cited our work.

[1] Luo, G, Blumenthal, M, Heide, M, Uecker, M. Bayesian MRI reconstruction with joint uncertainty estimation using diffusion models. Magn Reson Med. 2023; 1-17

[2] Blumenthal, M, Luo, G, Schilling, M, Holme, HCM, Uecker, M. Deep, deep learning with BART. Magn Reson Med. 2023; 89: 678- 693.

[3] Luo, G, Zhao, N, Jiang, W, Hui, ES, Cao, P. MRI reconstruction using deep Bayesian estimation. Magn Reson Med. 2020; 84: 2246–2261.

