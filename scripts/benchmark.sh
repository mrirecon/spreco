set -e

export TF_CPP_MIN_LOG_LEVEL=3
root_path=/home/gluo/spreco

recon()
{
    python prepare.py --workspace=$1 --h5_path=$2 --config=$3

    for file in $1/und_ksp_*.hdr; do
        tmp="${file##*/}"
        kspace="${tmp%.*}"
        idx=$(echo ${kspace} | awk -F'[._]' '{print $3}')
        mask="mask_${idx}"
        bart transpose 2 3 $1/$kspace $1/tmp
        bart ecalib -r20 -m1 -c0.0001 $1/tmp $1/coilsen
        bart pics -d5 -l1 -r 0.01 $1/tmp $1/coilsen $1/l1_recon_${idx}
        python recon/benchmark.py --config=$3 --workspace=$1 --kspace=$1/$kspace --mask=$1/$mask
    done
}

evaluate()
{
    for file in $1/mmse_*.hdr; do
        tmp="${file##*/}"
        mmse="${tmp%.*}"
        idx=$(echo ${mmse} | awk -F'[._]' '{print $2}')
        rss="rss_${idx}"
        l1_recon="l1_recon_${idx}"
        metric=$(python recon/benchmark.py --metric --mmse=$1/$mmse --ground=$1/$rss)
        echo "mmse->${metric}"
        metric=$(python recon/benchmark.py --metric --mmse=$1/$l1_recon --ground=$1/$rss)
        echo "l1->${metric}"
    done
}

# run reconstruction
config=$root_path/config_exp/configs/benchmark_1.yaml
res_folder=/home/gluo/workspace/sampling_posterior/benchmark/new_rep

raw_kspace=/home/ague/data/gluo/nyu_dataset/brain/multicoil_train/file_brain_AXFLAIR_200_6002631.h5
save_folder=$res_folder/subject_3_2
mkdir -p $save_folder
recon $save_folder $raw_kspace $config
evaluate $save_folder > $save_folder/info

raw_kspace=/home/ague/data/gluo/nyu_dataset/brain/multicoil_train/file_brain_AXFLAIR_209_6001428.h5
save_folder=$res_folder/subject_2_2
mkdir -p $save_folder
recon $save_folder $raw_kspace $config
evaluate $save_folder > $save_folder/info

raw_kspace=/home/ague/data/gluo/nyu_dataset/brain/multicoil_train/file_brain_AXFLAIR_200_6002566.h5
save_folder=$res_folder/subject_1_2
mkdir -p $save_folder
recon $save_folder $raw_kspace $config
evaluate $save_folder > $save_folder/info



echo "============="
config=$root_path/config_exp/configs/benchmark_2.yaml

raw_kspace=/home/ague/data/gluo/nyu_dataset/brain/multicoil_train/file_brain_AXFLAIR_200_6002631.h5
save_folder=$res_folder/subject_3_1
mkdir -p $save_folder
recon $save_folder $raw_kspace $config
evaluate $save_folder > $save_folder/info

raw_kspace=/home/ague/data/gluo/nyu_dataset/brain/multicoil_train/file_brain_AXFLAIR_209_6001428.h5
save_folder=$res_folder/subject_2_1
mkdir -p $save_folder
recon $save_folder $raw_kspace $config
evaluate $save_folder > $save_folder/info

raw_kspace=/home/ague/data/gluo/nyu_dataset/brain/multicoil_train/file_brain_AXFLAIR_200_6002566.h5
save_folder=$res_folder/subject_1_1
mkdir -p $save_folder
recon $save_folder $raw_kspace $config
evaluate $save_folder > $save_folder/info