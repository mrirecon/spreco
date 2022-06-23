# MRI Reconstruction via Data Driven Markov Chain with Joint Uncertainty Estimation (SPRECO)

This package is to reproduce the results in [Luo, et al (2022)](http://arxiv.org/abs/2202.01479). All programs are tested on Debian. There are two ways to set up the environment and please use python 3.8.

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

1. Sample the posterior [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuyuluo/spreco/blob/main/scripts/demo_recon.ipynb)
2. Train an image prior [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuyuluo/spreco/blob/main/scripts/demo_train.ipynb)

## Usage
Be careful with the files' location such as config files, training files and so on.
Before executing all the reconstruction scripts, please make sure that the reconstruction toolbox [BART](https://mrirecon.github.io/bart/) has been installed properly. If not, these tutorials [1](https://github.com/mrirecon/bart-workshop/blob/master/ismrm2021/bart_on_colab/colab_gpu_tutorial.ipynb), [2](https://github.com/mrirecon/bart-workshop/blob/master/doc/quick-install.md) might be helpful for you.

Explanation for files in the folder `scripts`
1. `demo_recon.ipynb` is an example for the application of the pre-trained prior to MRI reconstruction.
1. `demo_train.ipynb` is an example how to train a prior from images.
1. `preprocess_nyu.py` is the script to preprocess fastMRI dataset for training.
1. `prepare.py` is the script to prepare undersampled k-space with the subject from fastMRI dataset for the benchmark.
1. `benchmark.sh` is the script to perform benchmark.
1. `run_expr.sh` is the script to perform all experiments.
1. `recon_config.yaml` is the reconstruction config file for `demo_recon.ipynb`.

Remember to activate your environment if you're using conda. Please clone this repository before training and inferencing.

```shell
$ git clone https://github.com/mrirecon/spreco.git
$ cd spreco
```

### 1. Train priors

This is [explanation](#explanation-of-the-config-file-for-training) for the training configuration file. Start the trainer from the terminal

   ```shell
   $ python scripts/train/run_ncsn.py --train --config=config_exp/ncsn.yaml
   ```

After the training started, use tensorboard to track the training loss.

   ```shell
   $ tensorboard --logdir=$log_folder
   ```

### 2. Apply priors to MRI image reconstruction

This is the [explanation](#explanation-of-the-config-file-for-reconstruction) for the reconstruction configuration file. All experiments are listed below.

   1. Single coil unfolding
   2. Multi-coil reconstruction
   3. Increase the number of noise scales
   4. Extend iteration with MAP
   5. Investigate into burn-in phase
   6. Distortion</p>

We suggest you to go over the configuration files and check all the paths for trained model, k-space and workspace to save results before starting each experiment in `run_expr.sh`.