import shutil
import os

import numpy as np
from numpy.fft import fftshift, ifftshift, ifft2

import h5py
from spreco.common import utils
from tqdm import tqdm

data_path = '/home/ague/data/gluo/nyu_dataset/brain/multicoil_train'

"""
The nyu brain dataset has four types of constrast: AXT1POST, AXT2, AXT1 and AXFLAIR
"""

#contrast = 'AXT1POST'
#image_shape = [320, 320] # 949*16 = 15184

#contrast = 'AXT2'
#image_shape = [384, 384] #2678*16=42848

#contrast = 'AXT1'
#image_shape = [320, 320] #1447*16=23152

contrast = 'AXFLAIR'
image_shape = [320, 320] #344*16=5504

files_list = utils.find_files(data_path, '*%s_*.h5'%contrast)

savepath = "/home/ague/data/gluo/dataset/brain_mat_nyu/tmp"

def write_rss(filepath, savepath, file_id):
    try:
        fs = h5py.File(filepath, "r+")
        kspace = fs['kspace']
    except:
        print(filepath+' is corrupted')
        return file_id
    kspace = np.transpose(kspace, [2,3,1,0])
    coil_imgs = fftshift(ifft2(ifftshift(kspace), axes=(0,1)))

    coilsens = np.zeros_like(coil_imgs, dtype='complex64')
    
    if kspace.shape[0] == image_shape[0]*2 and kspace.shape[1] >= image_shape[1]:
        utils.log_to(savepath+'logs'+contrast, [filepath, kspace.shape])
        for i in range(kspace.shape[-1]):
            s_kspace = kspace[..., i]
            coilsens[..., i] = utils.bart(1, 'ecalib -m1 -r20 -c0.00001', s_kspace[np.newaxis, ...])
            #coilsens[..., i] = utils.bart(1, 'caldir 30', s_kspace[np.newaxis, ...])

        rss = np.squeeze(np.sum(coil_imgs*np.conj(coilsens), axis=2))
        rss = utils.bart(1, 'resize -c 0 '+str(image_shape[0])+ ' 1 ' +str(image_shape[1]), rss)

        for i in range(kspace.shape[-1]):
            tmp = rss[..., i]
            assert np.prod(tmp.shape) == np.prod(image_shape)
            np.savez(os.path.join(savepath,"nyu_"+contrast+'_ecalib_'+str(file_id)), rss=tmp)
            #utils.save_img(abs(tmp), os.path.join(savepath,"nyu_"+contrast+'_'+str(file_id)), np.min(abs(tmp)), np.max(abs(tmp)))
            file_id = file_id+1

        fs.close()
        return file_id
    else:
        return file_id

if False:
    file_id = 100000
    index = 0

    for i in tqdm(range(len(files_list))):
        file_id = write_rss(files_list[i], savepath, file_id)
        index = index + 1

if True:

    mat_files = sorted(utils.find_files(savepath, "nyu_%s_1*.npz"%contrast))
    order_arr = np.arange(len(mat_files))
    
    nr_total_files = len(order_arr) 
    split_pos = int(nr_total_files*0.9)
    train_files = []
    test_files = []


    for i in tqdm(range(split_pos)):
        train_files.append(mat_files[order_arr[i]])
        filename = os.path.split(mat_files[order_arr[i]])[-1]
        shutil.copy(mat_files[order_arr[i]], os.path.join("/home/ague/data/gluo/dataset/brain_mat_nyu/train", filename))


    for i in tqdm(range(split_pos,nr_total_files)):
        test_files.append(mat_files[order_arr[i]])
        filename = os.path.split(mat_files[order_arr[i]])[-1]
        shutil.copy(mat_files[order_arr[i]], os.path.join("/home/ague/data/gluo/dataset/brain_mat_nyu/test", filename))

