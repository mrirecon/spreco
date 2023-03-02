# this script is to demostrate the transferability
# bracket, single quote, double quote in shell script
root_path=/home/gluo/github/spreco
dat_path=/home/ague/transfer/2022-04-27_MRT5_DCRD_0014
res_folder=/home/gluo/workspace/sampling_posterior/transferability

#####
dat_file=meas_MID00131_FID03194_t1_tse_dark_fluid_tra_p2.dat
recon_ground_t1=0
if [ 1 = $recon_ground_t1 ]
then
    bart twixread -A $dat_path/$dat_file $res_folder/tmp
    bart reshape $(bart bitmask 0 1 2) 2 320 320 $res_folder/tmp $res_folder/ksp
    bart avg $(bart bitmask 14) $res_folder/ksp $res_folder/ksp_full
    bart avg $(bart bitmask 0) $res_folder/ksp_full $res_folder/tmp
    bart reshape $(bart bitmask 0 1 2) 320 320 1 $res_folder/tmp $res_folder/ksp_full_320
    bart fft -i $(bart bitmask 0 1) $res_folder/ksp_full_320 $res_folder/tmp_imgs
    bart rss $(bart bitmask 3) $res_folder/tmp_imgs $res_folder/imgs
fi

recon_t1=0
if [ 1 = $recon_t1 ]; then
    for i in $(seq 0 26); do
        bart slice 13 $i  $res_folder/ksp_full_320 $res_folder/slice_$i
        path="ksp_path: ${res_folder}/slice_${i}.cfl"
        sed -i --expression "s|ksp_path.*|$path|" $root_path/config_exp/configs/transfer.yaml
        python $root_path/scripts/recon/recon_sde.py --config=$root_path/config_exp/configs/transfer.yaml
    done
fi

#####
dat_file=meas_MID00153_FID03216_t2_tse_dark_fluid_tra.dat
recon_ground_t2=0
if [ 1 = $recon_ground_t2 ]; then
    bart twixread -A $dat_path/$dat_file $res_folder/t2_tmp
    bart reshape $(bart bitmask 0 1 2) 2 320 320 $res_folder/t2_tmp $res_folder/t2_ksp
    bart avg $(bart bitmask 0) $res_folder/t2_ksp $res_folder/tmp
    bart reshape $(bart bitmask 0 1 2) 320 320 1 $res_folder/tmp $res_folder/t2_ksp_full_320
    bart fft -i $(bart bitmask 0 1) $res_folder/t2_ksp_full_320 $res_folder/tmp_imgs
    bart rss $(bart bitmask 3) $res_folder/tmp_imgs $res_folder/t2_imgs
fi

recon_t2=0
if [ 1 = $recon_t2 ]; then
    for i in $(seq 0 26); do
        bart slice 13 $i  $res_folder/t2_ksp_full_320 $res_folder/t2_slice_$i
        path="ksp_path: ${res_folder}/t2_slice_${i}.cfl"
        sed -i --expression "s|ksp_path.*|$path|" $root_path/config_exp/configs/transfer.yaml
        python $root_path/scripts/recon/recon_sde.py --config=$root_path/config_exp/configs/transfer.yaml
    done
fi