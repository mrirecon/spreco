# MRI <span style="color:red">RECO</span>nstruction by <span style="color:red">S</span>ampling <span style="color:red">P</span>osterior (SPRECO)

This package is to reproduce the results in this paper [(pdf)](http://arxiv.org/abs/2202.01479). All programs are tested on Debian. There are two ways to set up the environment and please use python 3.8.

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

1. Sample the posterior $p(\mathbf{x}|\mathbf{y})$   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xuyuluo/spreco/blob/main/scripts/demo_recon.ipynb)
2. Train an image prior [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mrirecon/bart-workshop/blob/master/ismrm2021/bart_tensorflow/bart_tf.ipynb)

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

Start all experiments from the terminal
```shell
$ bash scripts/recon/run_expr.sh
```

## Explanation of the config file for training

The configuration file consists of four parts: model, saving, data and gpu.
Prepare training files and create the data pipe that matches the input of the selected network, and specify the location of them in the file config.yaml.

```yaml
# model
model: 'NCSN'
batch_size: 10
input_shape: [256, 256, 2]
data_chns: 'CPLX'   # complex input 
lr: 0.0001          # learning rate
begin_sigma: 0.3    # sigma_max
end_sigma: 0.01     # sigma_min
anneal_power: 2.
nr_levels: 10       # N
affine_x: True
nonlinearity: 'elu' # activation function
nr_filters: 64      # base number for the number of filters

# saving
seed: 1234          # random seed
max_keep: 100
max_epochs: 1000
save_interval: 50   # take snapshot of model per 50
saved_name: test_brain
log_folder:         # location for saving models, and training logs

# data
train_data_path: /home/ague/data/gluo/dataset/brain_mat/train
test_data_path: /home/ague/data/gluo/dataset/brain_mat/test
pattern: "nyu_AXFLAIR_*.npz"    # all the files matching this name pattern will be loaded for training or testing.
num_prepare: 10
print_loss: True
parts: 
    # specify the components that are used to constructed a data pipe
    # load the data with key 'rss' into numpy complex array, then squeeze the array, then normalize it with its maximum magnitude, then represent the complex image (width,height,1) with the float array (width,height,2), then crop the float array into the specified shape
    - [{func: 'npz_loader', key: 'rss'}, {func: 'squeeze'}, {func: 'normalize_with_max'}, {func: 'slice_image', shape: [256, 256, 2]}] 
    # function to generate noise indices
    - [{func: 'randint', nr_levels: 10}]

# gpu
nr_gpu: 2       # number of gpus
gpu_id: '1,2'   # PCI_BUS_ID
```

## Explanation of the config file for reconstruction

The configuration file for reconstruction specifies the following options.

1. which model is used to construct transition kernel?
2. which sampling pattern is used?
3. how many samples will be drawn from the posterior?
4. where to store the results?
5. the values of K, N, $\lambda$ in the proposed algorithm?
6. whether to use burn-in phase, at which time point to be burned?
7. where is the k-space data?

```yaml
burn_in: false
burn_t: 0.5     # at which time point to be burn
c_steps: 5      # K in Algorithm 1
s_stepsize: 25  # $\lambda$ in Algorithm 1
st: 30          # N=100-st in Algorithm 1
cal: 20         # calibration region
compute_residual: true
poisson: true   # possion sampling pattern
fx: 1.5         # if possion sampling is used, acceleration along x direction
fy: 1.5         # if possion sampling is used, acceleration along y direction
sampling_rate: 0.2  # if possion samples is not used, use Gaussian sampling pattern instead
nr_samples: 10  # how many samples will be drawn
model_folder: /home/ague/archive/projects/2021/gluo/mcmc_recon/models/net2/20211007-232921
model_name: sde_brain_500
ksp_path: /home/ague/data/gluo/dataset/LI/LI, Dahui_T2S_1.npz
workspace: /home/gluo/workspace/sampling_posterior/more_noise_scales

# misc
target_snr: 1
gpu_id: '3'
```