import os
import os.path as osp  # 为了方便路径操作，将 os.path 重命名为 osp
from glob import glob  # glob 用于文件匹配，类似于 shell 中的通配符操作
from torch.utils.data import Dataset  # 从 PyTorch 导入 Dataset 类，用于定义自定义数据集
import numpy as np  # 导入 NumPy，用于数值计算
import torch  # 导入 PyTorch
from functools import partial  # 导入 partial 函数，用于部分参数绑定
import torch.nn.functional as F  # PyTorch 中的函数式接口库，用于各种张量操作和损失函数


# 定义自定义数据集类 CTDataset，继承自 PyTorch 的 Dataset
class CTDataset(Dataset):
    # 初始化函数，设置数据集参数，mode 决定是训练还是测试，其他参数根据具体数据集要求
    #test_id=9
    def __init__(self, dataset, mode, test_ids=[0,1,2,3,4,5,6,7,8,9], dose=5, context=True):
        self.mode = mode  # 模式：'train' 或 'test'
        self.context = context  # 是否使用上下文（多帧数据）
        print(dataset)  # 输出数据集名称

        # mayo_2016_sim 和 mayo_2016 数据集的处理
        if dataset in ['mayo_2016_sim', 'mayo_2016']:
            # 设置数据路径
            if dataset == 'mayo_2016_sim':
                data_root = './data_preprocess/gen_data/mayo_2016_sim_npy'
            elif dataset == 'mayo_2016':
                data_root = './data_preprocess/gen_data/mayo_2016_npy'

            patient_ids = [67, 96, 109, 143, 192, 286, 291, 310, 333, 506]  # 定义患者 ID 列表
            if mode == 'train':
                patient_ids = [pid for i, pid in enumerate(patient_ids) if i not in test_ids]
            elif mode == 'test':
                # 确保 test_ids 是整数列表
                if isinstance(test_ids, str):
                    test_ids = list(map(int, test_ids.split(',')))  # 解析字符串为整数列表
                patient_ids = [patient_ids[i] for i in test_ids]

            # 处理目标图像路径
            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(glob(osp.join(data_root, ('L{:03d}_target_'.format(id) + '*_img.npy'))))
                patient_lists = patient_lists + patient_list[1:len(patient_list) - 1]  # 去掉首尾帧
            base_target = patient_lists  # 目标图像列表

            # 处理输入图像路径
            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(glob(osp.join(data_root, ('L{:03d}_{}_'.format(id, dose) + '*_img.npy'))))
                if context:  # 如果使用上下文帧
                    cat_patient_list = []
                    for i in range(1, len(patient_list) - 1):
                        patient_path = ''
                        for j in range(-1, 2):  # 将相邻帧拼接在一起
                            patient_path = patient_path + '~' + patient_list[i + j]
                        cat_patient_list.append(patient_path)
                    patient_lists = patient_lists + cat_patient_list
                else:  # 不使用上下文帧，只使用当前帧
                    patient_list = patient_list[1:len(patient_list) - 1]
                    patient_lists = patient_lists + patient_list
            base_input = patient_lists  # 输入图像列表

        # 处理 spect_10s1s 数据集
        elif dataset == 'spect_10s1s':
            data_root = './data_preprocess/gen_data/spect_10s1s_npy'
            patient_ids = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                           '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                           '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
                           '31', '32', '33', '34', '35', '36', '37', '38', '39', '40',
                           '41', '42', '43', '44', '45', '46', '47', '48', '49', '50']  # 定义患者 ID 列表
            if mode == 'train':
                patient_ids = [pid for i, pid in enumerate(patient_ids) if i not in test_ids]
            elif mode == 'test':
                # 确保 test_ids 是整数列表
                if isinstance(test_ids, str):
                    test_ids = list(map(int, test_ids.split(',')))  # 解析字符串为整数列表
                patient_ids = [patient_ids[i] for i in test_ids]

            # 处理目标图像路径
            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(glob(osp.join(data_root, ('P{}_target_'.format(id) + '*_img.npy'))))
                patient_lists = patient_lists + patient_list[1:len(patient_list) - 1]
            base_target = patient_lists  # 目标图像列表

            # 处理输入图像路径
            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(glob(osp.join(data_root, ('P{}_{}_'.format(id, dose) + '*_img.npy'))))
                if context:  # 使用上下文帧
                    cat_patient_list = []
                    for i in range(1, len(patient_list) - 1):
                        patient_path = ''
                        for j in range(-1, 2):
                            patient_path = patient_path + '~' + patient_list[i + j]
                        cat_patient_list.append(patient_path)
                    patient_lists = patient_lists + cat_patient_list
                else:
                    patient_list = patient_list[1:len(patient_list) - 1]
                    patient_lists = patient_lists + patient_list
                base_input = patient_lists  # 输入图像列表    
        # 处理 spect_10s2s 数据集
        elif dataset == 'spect_10s2s':
            data_root = './data_preprocess/gen_data/spect_10s2s_npy'
            patient_ids = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                           '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                           '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
                           '31', '32', '33', '34', '35', '36', '37', '38', '39', '40',
                           '41', '42', '43', '44', '45', '46', '47', '48', '49', '50']  # 定义患者 ID 列表
            if mode == 'train':
                patient_ids = [pid for i, pid in enumerate(patient_ids) if i not in test_ids]
            elif mode == 'test':
                # 确保 test_ids 是整数列表
                if isinstance(test_ids, str):
                    test_ids = list(map(int, test_ids.split(',')))  # 解析字符串为整数列表
                patient_ids = [patient_ids[i] for i in test_ids]

            # 处理目标图像路径
            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(glob(osp.join(data_root, ('P{}_target_'.format(id) + '*_img.npy'))))
                patient_lists = patient_lists + patient_list[1:len(patient_list) - 1]
            base_target = patient_lists  # 目标图像列表

            # 处理输入图像路径
            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(glob(osp.join(data_root, ('P{}_{}_'.format(id, dose) + '*_img.npy'))))
                if context:  # 使用上下文帧
                    cat_patient_list = []
                    for i in range(1, len(patient_list) - 1):
                        patient_path = ''
                        for j in range(-1, 2):
                            patient_path = patient_path + '~' + patient_list[i + j]
                        cat_patient_list.append(patient_path)
                    patient_lists = patient_lists + cat_patient_list
                else:
                    patient_list = patient_list[1:len(patient_list) - 1]
                    patient_lists = patient_lists + patient_list
                base_input = patient_lists  # 输入图像列表
        # 处理 spect_10s3s 数据集
        elif dataset == 'spect_10s3s':
            data_root = './data_preprocess/gen_data/spect_10s3s_npy'
            patient_ids = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                           '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                           '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
                           '31', '32', '33', '34', '35', '36', '37', '38', '39', '40',
                           '41', '42', '43', '44', '45', '46', '47', '48', '49', '50']  # 定义患者 ID 列表
            if mode == 'train':
                patient_ids = [pid for i, pid in enumerate(patient_ids) if i not in test_ids]
            elif mode == 'test':
                # 确保 test_ids 是整数列表
                if isinstance(test_ids, str):
                    test_ids = list(map(int, test_ids.split(',')))  # 解析字符串为整数列表
                patient_ids = [patient_ids[i] for i in test_ids]

            # 处理目标图像路径
            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(glob(osp.join(data_root, ('P{}_target_'.format(id) + '*_img.npy'))))
                patient_lists = patient_lists + patient_list[1:len(patient_list) - 1]
            base_target = patient_lists  # 目标图像列表

            # 处理输入图像路径
            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(glob(osp.join(data_root, ('P{}_{}_'.format(id, dose) + '*_img.npy'))))
                if context:  # 使用上下文帧
                    cat_patient_list = []
                    for i in range(1, len(patient_list) - 1):
                        patient_path = ''
                        for j in range(-1, 2):
                            patient_path = patient_path + '~' + patient_list[i + j]
                        cat_patient_list.append(patient_path)
                    patient_lists = patient_lists + cat_patient_list
                else:
                    patient_list = patient_list[1:len(patient_list) - 1]
                    patient_lists = patient_lists + patient_list
                base_input = patient_lists  # 输入图像列表    
        # 处理 spect_10s5s 数据集
        elif dataset == 'spect_10s5s':
            data_root = './data_preprocess/gen_data/spect_10s5s_npy'
            patient_ids = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                           '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                           '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
                           '31', '32', '33', '34', '35', '36', '37', '38', '39', '40',
                           '41', '42', '43', '44', '45', '46', '47', '48', '49', '50']  # 定义患者 ID 列表
            if mode == 'train':
                patient_ids = [pid for i, pid in enumerate(patient_ids) if i not in test_ids]
            elif mode == 'test':
                # 确保 test_ids 是整数列表
                if isinstance(test_ids, str):
                    test_ids = list(map(int, test_ids.split(',')))  # 解析字符串为整数列表
                patient_ids = [patient_ids[i] for i in test_ids]

            # 处理目标图像路径
            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(glob(osp.join(data_root, ('P{}_target_'.format(id) + '*_img.npy'))))
                patient_lists = patient_lists + patient_list[1:len(patient_list) - 1]
            base_target = patient_lists  # 目标图像列表

            # 处理输入图像路径
            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(glob(osp.join(data_root, ('P{}_{}_'.format(id, dose) + '*_img.npy'))))
                if context:  # 使用上下文帧
                    cat_patient_list = []
                    for i in range(1, len(patient_list) - 1):
                        patient_path = ''
                        for j in range(-1, 2):
                            patient_path = patient_path + '~' + patient_list[i + j]
                        cat_patient_list.append(patient_path)
                    patient_lists = patient_lists + cat_patient_list
                else:
                    patient_list = patient_list[1:len(patient_list) - 1]
                    patient_lists = patient_lists + patient_list
                base_input = patient_lists  # 输入图像列表
                
        # 处理 cgspect_8cg 数据集
        elif dataset == 'spect_8cg':
            data_root = './data_preprocess/gen_data/spect_8cg_npy'
            patient_ids = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010',
                            '011', '012', '013', '014', '015', '016', '017', '018', '019', '020',
                            '021', '022', '023', '024', '025', '026', '027', '028', '029', '030',
                            '031', '032', '033', '034', '035', '036', '037', '038', '039', '040',
                            '041', '042', '043', '044', '045', '046', '047', '048', '049', '050',
                            '051', '052', '053', '054', '055', '056', '057', '058', '059', '060',
                            '061', '062', '063', '064', '065', '066', '067', '068', '069', '070',
                            '071', '072', '073', '074', '075', '076', '077', '078', '079', '080',
                            '081', '082', '083', '084', '085', '086', '087', '088', '089', '090',
                            '091', '092', '093', '094', '095', '096', '097', '098', '099', '100',
                            '101', '102', '103', '104', '105', '106', '107', '108', '109', '110',
                            '111', '112', '113', '114', '115', '116', '117', '118', '119', '120',
                            '121', '122', '123', '124', '125', '126', '127', '128', '129', '130',
                            '131', '132', '133', '134', '135', '136', '137', '138', '139', '140',
                            '141', '142', '143', '144', '145', '146', '147', '148', '149', '150',
                            '151', '152', '153', '154', '155', '156', '157', '158', '159', '160',
                            '161', '162', '163', '164', '165', '166', '167', '168', '169', '170',
                            '171', '172', '173', '174', '175', '176', '177', '178', '179', '180',
                            '181', '182', '183', '184', '185', '186', '187', '188', '189', '190',
                            '191', '192', '193', '194', '195', '196', '197', '198', '199', '200',
                            '201', '202', '203', '204', '205', '206', '207', '208', '209', '210',
                            '211', '212', '213', '214', '215', '216', '217', '218', '219', '220',
                            '221', '222', '223', '224', '225', '226', '227', '228', '229', '230',
                            '231', '232', '233', '234', '235', '236', '237', '238', '239', '240',
                            '241', '242', '243', '244', '245', '246', '247', '248', '249', '250',
                            '251', '252', '253', '254', '255', '256', '257', '258', '259', '260',
                            '261', '262', '263', '264', '265', '266', '267', '268', '269', '270',
                            '271', '272', '273', '274', '275', '276', '277', '278', '279', '280',
                            '281', '282', '283', '284', '285', '286', '287', '288', '289', '290',
                            '291', '292', '293', '294', '295', '296', '297', '298', '299', '300',
                            '301', '302', '303', '304', '305', '306', '307', '308', '309', '310',
                            '311', '312', '313', '314', '315', '316', '317', '318', '319', '320',
                            '321', '322', '323', '324', '325', '326', '327', '328', '329', '330',
                            '331', '332', '333', '334', '335', '336', '337', '338', '339', '340',
                            '341', '342', '343', '344', '345', '346', '347', '348', '349', '350',
                            '351', '352', '353', '354', '355', '356', '357', '358', '359', '360',
                            '361', '362', '363', '364', '365', '366', '367', '368', '369', '370',
                            '371', '372', '373', '374', '375', '376', '377', '378', '379', '380',
                            '381', '382', '383', '384', '385', '386', '387', '388', '389', '390',
                            '391', '392', '393', '394', '395', '396', '397', '398', '399', '400']  # 定义患者 ID 列表
            if mode == 'train':
                patient_ids = [pid for i, pid in enumerate(patient_ids) if i not in test_ids]
            elif mode == 'test':
                # 确保 test_ids 是整数列表
                if isinstance(test_ids, str):
                    test_ids = list(map(int, test_ids.split(',')))  # 解析字符串为整数列表
                patient_ids = [patient_ids[i] for i in test_ids]

            # 处理目标图像路径
            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(glob(osp.join(data_root, ('P{}_target_'.format(id) + '*_img.npy'))))
                patient_lists = patient_lists + patient_list[1:len(patient_list) - 1]
            base_target = patient_lists  # 目标图像列表

            # 处理输入图像路径
            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(glob(osp.join(data_root, ('P{}_{}_'.format(id, dose) + '*_img.npy'))))
                if context:  # 使用上下文帧
                    cat_patient_list = []
                    for i in range(1, len(patient_list) - 1):
                        patient_path = ''
                        for j in range(-1, 2):
                            patient_path = patient_path + '~' + patient_list[i + j]
                        cat_patient_list.append(patient_path)
                    patient_lists = patient_lists + cat_patient_list
                else:
                    patient_list = patient_list[1:len(patient_list) - 1]
                    patient_lists = patient_lists + patient_list
                base_input = patient_lists  # 输入图像列表
                
        # 处理 cgspect_16cg 数据集
        elif dataset == 'spect_16cg':
            data_root = './data_preprocess/gen_data/spect_16cg_npy'
            patient_ids = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010',
                            '011', '012', '013', '014', '015', '016', '017', '018', '019', '020',
                            '021', '022', '023', '024', '025', '026', '027', '028', '029', '030',
                            '031', '032', '033', '034', '035', '036', '037', '038', '039', '040',
                            '041', '042', '043', '044', '045', '046', '047', '048', '049', '050',
                            '051', '052', '053', '054', '055', '056', '057', '058', '059', '060',
                            '061', '062', '063', '064', '065', '066', '067', '068', '069', '070',
                            '071', '072', '073', '074', '075', '076', '077', '078', '079', '080',
                            '081', '082', '083', '084', '085', '086', '087', '088', '089', '090',
                            '091', '092', '093', '094', '095', '096', '097', '098', '099', '100',
                            '101', '102', '103', '104', '105', '106', '107', '108', '109', '110',
                            '111', '112', '113', '114', '115', '116', '117', '118', '119', '120',
                            '121', '122', '123', '124', '125', '126', '127', '128', '129', '130',
                            '131', '132', '133', '134', '135', '136', '137', '138', '139', '140',
                            '141', '142', '143', '144', '145', '146', '147', '148', '149', '150',
                            '151', '152', '153', '154', '155', '156', '157', '158', '159', '160',
                            '161', '162', '163', '164', '165', '166', '167', '168', '169', '170',
                            '171', '172', '173', '174', '175', '176', '177', '178', '179', '180',
                            '181', '182', '183', '184', '185', '186', '187', '188', '189', '190',
                            '191', '192', '193', '194', '195', '196', '197', '198', '199', '200',
                            '201', '202', '203', '204', '205', '206', '207', '208', '209', '210',
                            '211', '212', '213', '214', '215', '216', '217', '218', '219', '220',
                            '221', '222', '223', '224', '225', '226', '227', '228', '229', '230',
                            '231', '232', '233', '234', '235', '236', '237', '238', '239', '240',
                            '241', '242', '243', '244', '245', '246', '247', '248', '249', '250',
                            '251', '252', '253', '254', '255', '256', '257', '258', '259', '260',
                            '261', '262', '263', '264', '265', '266', '267', '268', '269', '270',
                            '271', '272', '273', '274', '275', '276', '277', '278', '279', '280',
                            '281', '282', '283', '284', '285', '286', '287', '288', '289', '290',
                            '291', '292', '293', '294', '295', '296', '297', '298', '299', '300',
                            '301', '302', '303', '304', '305', '306', '307', '308', '309', '310',
                            '311', '312', '313', '314', '315', '316', '317', '318', '319', '320',
                            '321', '322', '323', '324', '325', '326', '327', '328', '329', '330',
                            '331', '332', '333', '334', '335', '336', '337', '338', '339', '340',
                            '341', '342', '343', '344', '345', '346', '347', '348', '349', '350',
                            '351', '352', '353', '354', '355', '356', '357', '358', '359', '360',
                            '361', '362', '363', '364', '365', '366', '367', '368', '369', '370',
                            '371', '372', '373', '374', '375', '376', '377', '378', '379', '380',
                            '381', '382', '383', '384', '385', '386', '387', '388', '389', '390',
                            '391', '392', '393', '394', '395', '396', '397', '398', '399', '400',
                            '401', '402', '403', '404', '405', '406', '407', '408', '409', '410', 
                           '411', '412', '413', '414', '415', '416', '417', '418', '419', '420', 
                           '421', '422', '423', '424', '425', '426', '427', '428', '429', '430', 
                           '431', '432', '433', '434', '435', '436', '437', '438', '439', '440', 
                           '441', '442', '443', '444', '445', '446', '447', '448', '449', '450', 
                           '451', '452', '453', '454', '455', '456', '457', '458', '459', '460', 
                           '461', '462', '463', '464', '465', '466', '467', '468', '469', '470', 
                           '471', '472', '473', '474', '475', '476', '477', '478', '479', '480',
                           '481', '482', '483', '484', '485', '486', '487', '488', '489', '490', 
                           '491', '492', '493', '494', '495', '496', '497', '498', '499', '500',
                           '501', '502', '503', '504', '505', '506', '507', '508', '509', '510', 
                           '511', '512', '513', '514', '515', '516', '517', '518', '519', '520', 
                           '521', '522', '523', '524', '525', '526', '527', '528', '529', '530',
                           '531', '532', '533', '534', '535', '536', '537', '538', '539', '540', 
                           '541', '542', '543', '544', '545', '546', '547', '548', '549', '550',
                           '551', '552', '553', '554', '555', '556', '557', '558', '559', '560', 
                           '561', '562', '563', '564', '565', '566', '567', '568', '569', '570',
                           '571', '572', '573', '574', '575', '576', '577', '578', '579', '580',
                           '581', '582', '583', '584', '585', '586', '587', '588', '589', '590', 
                           '591', '592', '593', '594', '595', '596', '597', '598', '599', '600',
                           '601', '602', '603', '604', '605', '606', '607', '608', '609', '610',
                           '611', '612', '613', '614', '615', '616', '617', '618', '619', '620',
                           '621', '622', '623', '624', '625', '626', '627', '628', '629', '630', 
                           '631', '632', '633', '634', '635', '636', '637', '638', '639', '640', 
                           '641', '642', '643', '644', '645', '646', '647', '648', '649', '650', 
                           '651', '652', '653', '654', '655', '656', '657', '658', '659', '660', 
                           '661', '662', '663', '664', '665', '666', '667', '668', '669', '670', 
                           '671', '672', '673', '674', '675', '676', '677', '678', '679', '680', 
                           '681', '682', '683', '684', '685', '686', '687', '688', '689', '690', 
                           '691', '692', '693', '694', '695', '696', '697', '698', '699', '700', 
                           '701', '702', '703', '704', '705', '706', '707', '708', '709', '710', 
                           '711', '712', '713', '714', '715', '716', '717', '718', '719', '720',
                           '721', '722', '723', '724', '725', '726', '727', '728', '729', '730',
                           '731', '732', '733', '734', '735', '736', '737', '738', '739', '740', 
                           '741', '742', '743', '744', '745', '746', '747', '748', '749', '750', 
                           '751', '752', '753', '754', '755', '756', '757', '758', '759', '760',
                           '761', '762', '763', '764', '765', '766', '767', '768', '769', '770', 
                           '771', '772', '773', '774', '775', '776', '777', '778', '779', '780', 
                           '781', '782', '783', '784', '785', '786', '787', '788', '789', '790', 
                           '791', '792', '793', '794', '795', '796', '797', '798', '799', '800']  # 定义患者 ID 列表
            if mode == 'train':
                patient_ids = [pid for i, pid in enumerate(patient_ids) if i not in test_ids]
            elif mode == 'test':
                # 确保 test_ids 是整数列表
                if isinstance(test_ids, str):
                    test_ids = list(map(int, test_ids.split(',')))  # 解析字符串为整数列表
                patient_ids = [patient_ids[i] for i in test_ids]

            # 处理目标图像路径
            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(glob(osp.join(data_root, ('P{}_target_'.format(id) + '*_img.npy'))))
                patient_lists = patient_lists + patient_list[1:len(patient_list) - 1]
            base_target = patient_lists  # 目标图像列表

            # 处理输入图像路径
            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(glob(osp.join(data_root, ('P{}_{}_'.format(id, dose) + '*_img.npy'))))
                if context:  # 使用上下文帧
                    cat_patient_list = []
                    for i in range(1, len(patient_list) - 1):
                        patient_path = ''
                        for j in range(-1, 2):
                            patient_path = patient_path + '~' + patient_list[i + j]
                        cat_patient_list.append(patient_path)
                    patient_lists = patient_lists + cat_patient_list
                else:
                    patient_list = patient_list[1:len(patient_list) - 1]
                    patient_lists = patient_lists + patient_list
                base_input = patient_lists  # 输入图像列表



        self.input = base_input  # 输入数据
        self.target = base_target  # 目标数据
        print(len(self.input))  # 打印输入数据数量
        print(len(self.target))  # 打印目标数据数量

    def __getitem__(self, index):
        # 根据索引获取对应的输入和目标数据
        input, target = self.input[index], self.target[index]

        if self.context:  # 如果使用上下文
            input = input.split('~')  # 按符号分割帧
            inputs = []
            for i in range(1, len(input)):
                inputs.append(np.load(input[i])[np.newaxis, ...].astype(np.float32))  # 加载并扩展维度
            input = np.concatenate(inputs, axis=0)  # 合并成一个 3 通道的张量
        else:
            input = np.load(input)[np.newaxis, ...].astype(np.float32)  # 加载单帧数据
        target = np.load(target)[np.newaxis, ...].astype(np.float32)  # 加载目标数据
        
        NMSE_scale = np.sum(target) / np.sum(input)
        input = input * NMSE_scale
        
        input = self.normalize_(input)  # 归一化输入数据
        target = self.normalize_(target)  # 归一化目标数据

        return input, target  # 返回输入和目标

    def __len__(self):
        # 返回数据集长度
        return len(self.target)

    # def normalize_(self, img, MIN_B=0, MAX_B=65535):
    #     """
    #     将 SPECT 图像归一化到 [0, 1] 的范围。
    #
    #     :param img: 输入图像
    #     :param MIN_B: 图像的最小物理值 (SPECT 的 Minimum)
    #     :param MAX_B: 图像的最大物理值 (SPECT 的 Maximum)
    #     :return: 归一化后的图像
    #     """
    #     img[img < MIN_B] = MIN_B  # 限制图像最小值
    #     img[img > MAX_B] = MAX_B  # 限制图像最大值
    #     img = (img - MIN_B) / (MAX_B - MIN_B)  # 归一化到 [0, 1]
    #     return img
    def normalize_(self, img):
        MIN_B = np.min(img)  # 动态计算图像的最小值
        MAX_B = np.max(img)  # 动态计算图像的最大值
        img[img < MIN_B] = MIN_B
        img[img > MAX_B] = MAX_B
        return (img - MIN_B) / (MAX_B - MIN_B)  # 归一化到 [0, 1]


# 定义数据集字典，根据不同的情况选择合适的数据集加载方式
dataset_dict = {
    #'train': partial(CTDataset, dataset='mayo_2016_sim', mode='train', test_id=9, dose=5, context=True),
    'mayo_2016_sim': partial(CTDataset, dataset='mayo_2016_sim', mode='test', test_ids=[9], dose=5, context=True),
    'mayo_2016': partial(CTDataset, dataset='mayo_2016', mode='test', test_ids=[9], dose=25, context=True),
    'train': partial(CTDataset, dataset='spect_8cg', mode='train', test_ids=[0, 1, 2, 3, 4], dose='8cg', context=True),
    'spect_10s1s': partial(CTDataset, dataset='spect_10s1s', mode='test', test_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dose=10, context=True),
    'spect_10s2s': partial(CTDataset, dataset='spect_10s2s', mode='test', test_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dose=20, context=True),
    'spect_10s3s': partial(CTDataset, dataset='spect_10s3s', mode='test', test_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dose=30, context=True),
    'spect_10s5s': partial(CTDataset, dataset='spect_10s5s', mode='test', test_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dose=50, context=True),
    'spect_8cg': partial(CTDataset, dataset='spect_8cg', mode='test', test_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dose='8cg', context=True),
    'spect_16cg': partial(CTDataset, dataset='spect_16cg', mode='test', test_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dose='16cg', context=True),
}

