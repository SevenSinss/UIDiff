import os
import os.path as osp
import argparse
import numpy as np
from natsort import natsorted
from glob import glob
import pydicom

def save_dataset(args):
    if not osp.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    patient_ids = ['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09', 'P10',
                   'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20',
                   'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30',
                   'P31', 'P32', 'P33', 'P34', 'P35', 'P36', 'P37', 'P38', 'P39', 'P40',
                   'P41', 'P42', 'P43', 'P44', 'P45', 'P46', 'P47', 'P48', 'P49', 'P50']

    io = 'target'
    for p_ind, patient_id in enumerate(patient_ids):
        print(patient_id)
        if p_ind >= 0:
            patient_path = osp.join(args.data_path, patient_id, 'dose_10s')
            data_paths = natsorted(glob(osp.join(patient_path, '*.IMA')))
            for slice, data_path in enumerate(data_paths):
                im = pydicom.dcmread(data_path)  # 强制读取文件
                f = np.array(im.pixel_array)
                f_name = '{}_{}_{:0>3d}_img.npy'.format(patient_id, io, slice+1)
                np.save(osp.join(args.save_path, f_name), f.astype(np.uint16))

    io = '50'
    for p_ind, patient_id in enumerate(patient_ids):
        print(patient_id)
        if p_ind >= 0:
            patient_path = osp.join(args.data_path, patient_id, 'dose_5s')
            data_paths = natsorted(glob(osp.join(patient_path, '*.IMA')))
            for slice, data_path in enumerate(data_paths):
                im = pydicom.dcmread(data_path)  # 强制读取文件
                f = np.array(im.pixel_array)
                f_name = '{}_{}_{:0>3d}_img.npy'.format(patient_id, io, slice+1)
                np.save(osp.join(args.save_path, f_name), f.astype(np.uint16))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../../data_spect')   # data format: dicom
    parser.add_argument('--save_path', type=str, default='./gen_data/spect_10s5s_npy/')
    args = parser.parse_args()

    save_dataset(args)
