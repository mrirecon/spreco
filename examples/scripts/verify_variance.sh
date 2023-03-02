dat_path=/home/ague/transfer/2022-04-27_MRT5_DCRD_0014

res_folder=/home/gluo/workspace/sampling_posterior/verify_variance

file_1=meas_MID00135_FID03198_t2_tse_dark_fluid_tra.dat
file_2=meas_MID00137_FID03200_t2_tse_dark_fluid_tra.dat
file_3=meas_MID00139_FID03202_t2_tse_dark_fluid_tra.dat
file_4=meas_MID00141_FID03204_t2_tse_dark_fluid_tra.dat
file_5=meas_MID00143_FID03206_t2_tse_dark_fluid_tra.dat
file_6=meas_MID00145_FID03208_t2_tse_dark_fluid_tra.dat
file_7=meas_MID00147_FID03210_t2_tse_dark_fluid_tra.dat
file_8=meas_MID00149_FID03212_t2_tse_dark_fluid_tra.dat
file_9=meas_MID00151_FID03214_t2_tse_dark_fluid_tra.dat
file_10=meas_MID00153_FID03216_t2_tse_dark_fluid_tra.dat

for i in $(seq 1 10); do
    var="file_$i"
    bart twixread -A $dat_path/"${!var}" $res_folder/tmp
    bart reshape $(bart bitmask 0 1 2) 2 320 320 $res_folder/tmp $res_folder/ksp_$i
    bart fft -i $(bart bitmask 1 2) $res_folder/ksp_$i $res_folder/tmp_imgs
    bart rss $(bart bitmask 3) $res_folder/tmp_imgs $res_folder/imgs_v$i
done
