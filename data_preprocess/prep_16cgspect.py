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
                    'P391', 'P392', 'P393', 'P394', 'P395', 'P396', 'P397', 'P398', 'P399', 'P400',
                    'P401', 'P402', 'P403', 'P404', 'P405', 'P406', 'P407', 'P408', 'P409', 'P410',
                    'P411', 'P412', 'P413', 'P414', 'P415', 'P416', 'P417', 'P418', 'P419', 'P420', 
                    'P421', 'P422', 'P423', 'P424', 'P425', 'P426', 'P427', 'P428', 'P429', 'P430', 
                    'P431', 'P432', 'P433', 'P434', 'P435', 'P436', 'P437', 'P438', 'P439', 'P440',
                    'P441', 'P442', 'P443', 'P444', 'P445', 'P446', 'P447', 'P448', 'P449', 'P450', 
                    'P451', 'P452', 'P453', 'P454', 'P455', 'P456', 'P457', 'P458', 'P459', 'P460', 
                    'P461', 'P462', 'P463', 'P464', 'P465', 'P466', 'P467', 'P468', 'P469', 'P470', 
                    'P471', 'P472', 'P473', 'P474', 'P475', 'P476', 'P477', 'P478', 'P479', 'P480', 
                    'P481', 'P482', 'P483', 'P484', 'P485', 'P486', 'P487', 'P488', 'P489', 'P490', 
                    'P491', 'P492', 'P493', 'P494', 'P495', 'P496', 'P497', 'P498', 'P499', 'P500', 
                    'P501', 'P502', 'P503', 'P504', 'P505', 'P506', 'P507', 'P508', 'P509', 'P510', 
                    'P511', 'P512', 'P513', 'P514', 'P515', 'P516', 'P517', 'P518', 'P519', 'P520', 
                    'P521', 'P522', 'P523', 'P524', 'P525', 'P526', 'P527', 'P528', 'P529', 'P530', 
                    'P531', 'P532', 'P533', 'P534', 'P535', 'P536', 'P537', 'P538', 'P539', 'P540', 
                    'P541', 'P542', 'P543', 'P544', 'P545', 'P546', 'P547', 'P548', 'P549', 'P550', 
                    'P551', 'P552', 'P553', 'P554', 'P555', 'P556', 'P557', 'P558', 'P559', 'P560', 
                    'P561', 'P562', 'P563', 'P564', 'P565', 'P566', 'P567', 'P568', 'P569', 'P570', 
                    'P571', 'P572', 'P573', 'P574', 'P575', 'P576', 'P577', 'P578', 'P579', 'P580', 
                    'P581', 'P582', 'P583', 'P584', 'P585', 'P586', 'P587', 'P588', 'P589', 'P590', 
                    'P591', 'P592', 'P593', 'P594', 'P595', 'P596', 'P597', 'P598', 'P599', 'P600', 
                    'P601', 'P602', 'P603', 'P604', 'P605', 'P606', 'P607', 'P608', 'P609', 'P610', 
                    'P611', 'P612', 'P613', 'P614', 'P615', 'P616', 'P617', 'P618', 'P619', 'P620', 
                    'P621', 'P622', 'P623', 'P624', 'P625', 'P626', 'P627', 'P628', 'P629', 'P630', 
                    'P631', 'P632', 'P633', 'P634', 'P635', 'P636', 'P637', 'P638', 'P639', 'P640', 
                    'P641', 'P642', 'P643', 'P644', 'P645', 'P646', 'P647', 'P648', 'P649', 'P650', 
                    'P651', 'P652', 'P653', 'P654', 'P655', 'P656', 'P657', 'P658', 'P659', 'P660', 
                    'P661', 'P662', 'P663', 'P664', 'P665', 'P666', 'P667', 'P668', 'P669', 'P670', 
                    'P671', 'P672', 'P673', 'P674', 'P675', 'P676', 'P677', 'P678', 'P679', 'P680', 
                    'P681', 'P682', 'P683', 'P684', 'P685', 'P686', 'P687', 'P688', 'P689', 'P690', 
                    'P691', 'P692', 'P693', 'P694', 'P695', 'P696', 'P697', 'P698', 'P699', 'P700', 
                    'P701', 'P702', 'P703', 'P704', 'P705', 'P706', 'P707', 'P708', 'P709', 'P710',
                    'P711', 'P712', 'P713', 'P714', 'P715', 'P716', 'P717', 'P718', 'P719', 'P720', 
                    'P721', 'P722', 'P723', 'P724', 'P725', 'P726', 'P727', 'P728', 'P729', 'P730', 
                    'P731', 'P732', 'P733', 'P734', 'P735', 'P736', 'P737', 'P738', 'P739', 'P740', 
                    'P741', 'P742', 'P743', 'P744', 'P745', 'P746', 'P747', 'P748', 'P749', 'P750', 
                    'P751', 'P752', 'P753', 'P754', 'P755', 'P756', 'P757', 'P758', 'P759', 'P760', 
                    'P761', 'P762', 'P763', 'P764', 'P765', 'P766', 'P767', 'P768', 'P769', 'P770', 
                    'P771', 'P772', 'P773', 'P774', 'P775', 'P776', 'P777', 'P778', 'P779', 'P780', 
                    'P781', 'P782', 'P783', 'P784', 'P785', 'P786', 'P787', 'P788', 'P789', 'P790', 
                    'P791', 'P792', 'P793', 'P794', 'P795', 'P796', 'P797', 'P798', 'P799', 'P800']

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

    io = '16cg'
    for p_ind, patient_id in enumerate(patient_ids):
        print(patient_id)
        if p_ind >= 0:
            patient_path = osp.join(args.data_path, patient_id, '16cg')
            data_paths = natsorted(glob(osp.join(patient_path, '*.tif')))
            for slice, data_path in enumerate(data_paths):
                im = Image.open(data_path)
                f = np.array(im)
                f_name = '{}_{}_{:0>3d}_img.npy'.format(patient_id, io, slice+1)
                np.save(osp.join(args.save_path, f_name), f.astype(np.uint16))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../../cgspect_16cg')   # data format: tif
    parser.add_argument('--save_path', type=str, default='./gen_data/spect_16cg_npy/')
    args = parser.parse_args()

    save_dataset(args)