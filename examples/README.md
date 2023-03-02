# Overview
You are able to get code to generate the results in the paper title "Bayesian MRI reconstruction with joint uncertainty estimation using diffusion models" (10.1002/mrm.29624), using the code and config files here.

Be careful with the files' location such as config files, training files and so on.
Before executing all the reconstruction scripts, please make sure that the reconstruction toolbox [BART](https://mrirecon.github.io/bart/) has been installed properly. If not, these tutorials [1](https://github.com/mrirecon/bart-workshop/blob/master/ismrm2021/bart_on_colab/colab_gpu_tutorial.ipynb), [2](https://github.com/mrirecon/bart-workshop/blob/master/doc/quick-install.md) will be helpful for you.

## Information on files in the folder `scripts`
1. `demo_recon.ipynb` is an example for the application of the pre-trained prior to MRI reconstruction.
2. `demo_train.ipynb` is an example how to train a prior from images.
3. `preprocess_nyu.py` is the script to preprocess fastMRI dataset for training.
4. `prepare.py` is the script to prepare undersampled k-space with the subject from fastMRI dataset for the benchmark.
5. `benchmark.sh` is the script to perform benchmark.
6. `run_expr.sh` is the script to perform all experiments.
7. `recon_config.yaml` is the reconstruction config file for `demo_recon.ipynb`.

Remember to activate your environment if you're using conda. Please clone this repository before training and inferencing.

```shell
$ git clone https://github.com/mrirecon/spreco.git
$ cd spreco
```

## Train priors

Start the trainer from the terminal

   ```shell
   $ python scripts/train/run_ncsn.py --train --config=config_exp/ncsn.yaml
   ```

After the training started, use tensorboard to track the training loss.

   ```shell
   $ tensorboard --logdir=$log_folder
   ```