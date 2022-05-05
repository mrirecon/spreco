set -e

export TF_CPP_MIN_LOG_LEVEL=3
root_path=/home/gluo/spreco

## unfolding
python $root_path/scripts/recon/unfold_ncsn.py --config=$root_path/config_exp/configs/unfolding_1.yaml
python $root_path/scripts/recon/unfold_ncsn.py --config=$root_path/config_exp/configs/unfolding_2.yaml

## multi-coil
python $root_path/scripts/recon/recon_ncsn.py --config=$root_path/config_exp/configs/multi_coil.yaml

## more noise scales
python $root_path/scripts/recon/recon_sde.py --config=$root_path/config_exp/configs/more_noise_scales.yaml

## map at the end
python $root_path/scripts/recon/recon_sde.py --config=$root_path/config_exp/configs/map_end.yaml

## burn in phase
python $root_path/scripts/recon/recon_sde.py --config=$root_path/config_exp/configs/burn-in/10_config_0.1.yaml
python $root_path/scripts/recon/recon_sde.py --config=$root_path/config_exp/configs/burn-in/10_config_0.2.yaml
python $root_path/scripts/recon/recon_sde.py --config=$root_path/config_exp/configs/burn-in/10_config_0.3.yaml
python $root_path/scripts/recon/recon_sde.py --config=$root_path/config_exp/configs/burn-in/10_config_0.4.yaml
python $root_path/scripts/recon/recon_sde.py --config=$root_path/config_exp/configs/burn-in/10_config_0.5.yaml
python $root_path/scripts/recon/recon_sde.py --config=$root_path/config_exp/configs/burn-in/10_config.yaml
python $root_path/scripts/recon/recon_sde.py --config=$root_path/config_exp/configs/burn-in/20_config_0.1.yaml
python $root_path/scripts/recon/recon_sde.py --config=$root_path/config_exp/configs/burn-in/20_config_0.2.yaml
python $root_path/scripts/recon/recon_sde.py --config=$root_path/config_exp/configs/burn-in/20_config_0.3.yaml
python $root_path/scripts/recon/recon_sde.py --config=$root_path/config_exp/configs/burn-in/20_config_0.4.yaml
python $root_path/scripts/recon/recon_sde.py --config=$root_path/config_exp/configs/burn-in/20_config_0.5.yaml
python $root_path/scripts/recon/recon_sde.py --config=$root_path/config_exp/configs/burn-in/20_config.yaml

## overfit
python $root_path/scripts/recon/recon_sde.py --config=$root_path/config_exp/configs/overfit/config_1.yaml
python $root_path/scripts/recon/recon_sde.py --config=$root_path/config_exp/configs/overfit/config_2.yaml
python $root_path/scripts/recon/recon_sde.py --config=$root_path/config_exp/configs/overfit/config_3.yaml
python $root_path/scripts/recon/recon_sde.py --config=$root_path/config_exp/configs/overfit/config_4.yaml
python $root_path/scripts/recon/recon_sde.py --config=$root_path/config_exp/configs/overfit/config_5.yaml