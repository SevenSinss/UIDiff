import os
import os.path as osp
import argparse
import numpy as np
from natsort import natsorted
from glob import glob
from PIL import Image

def save_dataset(args):
    if not osp.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    patient_ids = ['P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008', 'P009', 'P010',
                    'P011', 'P012', 'P013', 'P014', 'P015', 'P016', 'P017', 'P018', 'P019', 'P020',
                    'P021', 'P022', 'P023', 'P024', 'P025', 'P026', 'P027', 'P028', 'P029', 'P030',
                    'P031', 'P032', 'P033', 'P034', 'P035', 'P036', 'P037', 'P038', 'P039', 'P040',
                    'P041', 'P042', 'P043', 'P044', 'P045', 'P046', 'P047', 'P048', 'P049', 'P050',
                    'P051', 'P052', 'P053', 'P054', 'P055', 'P056', 'P057', 'P058', 'P059', 'P060',
                    'P061', 'P062', 'P063', 'P064', 'P065', 'P066', 'P067', 'P068', 'P069', 'P070',
                    'P071', 'P072', 'P073', 'P074', 'P075', 'P076', 'P077', 'P078', 'P079', 'P080',
                    'P081', 'P082', 'P083', 'P084', 'P085', 'P086', 'P087', 'P088', 'P089', 'P090',
                    'P091', 'P092', 'P093', 'P094', 'P095', 'P096', 'P097', 'P098', 'P099', 'P100',
                    'P101', 'P102', 'P103', 'P104', 'P105', 'P106', 'P107', 'P108', 'P109', 'P110',
                    'P111', 'P112', 'P113', 'P114', 'P115', 'P116', 'P117', 'P118', 'P119', 'P120',
                    'P121', 'P122', 'P123', 'P124', 'P125', 'P126', 'P127', 'P128', 'P129', 'P130',
                    'P131', 'P132', 'P133', 'P134', 'P135', 'P136', 'P137', 'P138', 'P139', 'P140',
                    'P141', 'P142', 'P143', 'P144', 'P145', 'P146', 'P147', 'P148', 'P149', 'P150',
                    'P151', 'P152', 'P153', 'P154', 'P155', 'P156', 'P157', 'P158', 'P159', 'P160',
                    'P161', 'P162', 'P163', 'P164', 'P165', 'P166', 'P167', 'P168', 'P169', 'P170',
                    'P171', 'P172', 'P173', 'P174', 'P175', 'P176', 'P177', 'P178', 'P179', 'P180',
                    'P181', 'P182', 'P183', 'P184', 'P185', 'P186', 'P187', 'P188', 'P189', 'P190',
                    'P191', 'P192', 'P193', 'P194', 'P195', 'P196', 'P197', 'P198', 'P199', 'P200',
                    'P201', 'P202', 'P203', 'P204', 'P205', 'P206', 'P207', 'P208', 'P209', 'P210',
                    'P211', 'P212', 'P213', 'P214', 'P215', 'P216', 'P217', 'P218', 'P219', 'P220',
                    'P221', 'P222', 'P223', 'P224', 'P225', 'P226', 'P227', 'P228', 'P229', 'P230',
                    'P231', 'P232', 'P233', 'P234', 'P235', 'P236', 'P237', 'P238', 'P239', 'P240',
                    'P241', 'P242', 'P243', 'P244', 'P245', 'P246', 'P247', 'P248', 'P249', 'P250',
                    'P251', 'P252', 'P253', 'P254', 'P255', 'P256', 'P257', 'P258', 'P259', 'P260',
                    'P261', 'P262', 'P263', 'P264', 'P265', 'P266', 'P267', 'P268', 'P269', 'P270',
                    'P271', 'P272', 'P273', 'P274', 'P275', 'P276', 'P277', 'P278', 'P279', 'P280',
                    'P281', 'P282', 'P283', 'P284', 'P285', 'P286', 'P287', 'P288', 'P289', 'P290',
                    'P291', 'P292', 'P293', 'P294', 'P295', 'P296', 'P297', 'P298', 'P299', 'P300',
                    'P301', 'P302', 'P303', 'P304', 'P305', 'P306', 'P307', 'P308', 'P309', 'P310',
                    'P311', 'P312', 'P313', 'P314', 'P315', 'P316', 'P317', 'P318', 'P319', 'P320',
                    'P321', 'P322', 'P323', 'P324', 'P325', 'P326', 'P327', 'P328', 'P329', 'P330',
                    'P331', 'P332', 'P333', 'P334', 'P335', 'P336', 'P337', 'P338', 'P339', 'P340',
                    'P341', 'P342', 'P343', 'P344', 'P345', 'P346', 'P347', 'P348', 'P349', 'P350',
                    'P351', 'P352', 'P353', 'P354', 'P355', 'P356', 'P357', 'P358', 'P359', 'P360',
                    'P361', 'P362', 'P363', 'P364', 'P365', 'P366', 'P367', 'P368', 'P369', 'P370',
                    'P371', 'P372', 'P373', 'P374', 'P375', 'P376', 'P377', 'P378', 'P379', 'P380',
                    'P381', 'P382', 'P383', 'P384', 'P385', 'P386', 'P387', 'P388', 'P389', 'P390',
                    'P391', 'P392', 'P393', 'P394', 'P395', 'P396', 'P397', 'P398', 'P399', 'P400']

    io = 'target'
    for p_ind, patient_id in enumerate(patient_ids):
        print(patient_id)
        if p_ind >= 0:
            patient_path = osp.join(args.data_path, patient_id, 'corrsta')
            data_paths = natsorted(glob(osp.join(patient_path, '*.tif')))
            for slice, data_path in enumerate(data_paths):
                im = Image.open(data_path)
                f = np.array(im)
                f_name = '{}_{}_{:0>3d}_img.npy'.format(patient_id, io, slice+1)
                np.save(osp.join(args.save_path, f_name), f.astype(np.uint16))

    io = '8cg'
    for p_ind, patient_id in enumerate(patient_ids):
        print(patient_id)
        if p_ind >= 0:
            patient_path = osp.join(args.data_path, patient_id, '8cg')
            data_paths = natsorted(glob(osp.join(patient_path, '*.tif')))
            for slice, data_path in enumerate(data_paths):
                im = Image.open(data_path)
                f = np.array(im)
                f_name = '{}_{}_{:0>3d}_img.npy'.format(patient_id, io, slice+1)
                np.save(osp.join(args.save_path, f_name), f.astype(np.uint16))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../../cgspect_8cg')   # data format: tif
    parser.add_argument('--save_path', type=str, default='./gen_data/spect_8cg_npy/')
    args = parser.parse_args()

    save_dataset(args)