import os
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity
from sklearn.metrics import mean_squared_error
from datetime import datetime
from tifffile import imread


def normalize_image(img):
    """将图像归一化到 [0, 1] 范围"""
    img_min = np.min(img)
    img_max = np.max(img)
    if img_max == img_min:  # 避免除零
        return img - img_min
    return (img - img_min) / (img_max - img_min)


def data_show(full_dose_folder, gen_dose_folder, QA_save_path):
    # 患者 ID 列表
    patient_ids = [
        '001', '002', '004', '005', '011', '012', '018', '019', '022', '025',
        '051', '052', '054', '055', '061', '062', '068', '069', '072', '075',
        '101', '102', '104', '105', '111', '112', '118', '119', '122', '125',
        '151', '152', '154', '155', '161', '162', '168', '169', '172', '175',
        '201', '202', '204', '205', '211', '212', '218', '219', '222', '225',
        '251', '252', '254', '255', '261', '262', '268', '269', '272', '275',
        '301', '302', '304', '305', '311', '312', '318', '319', '322', '325',
        '351', '352', '354', '355', '361', '362', '368', '369', '372', '375'
    ]

    # 定义组别（10 组，每组 8 个患者）
    groups = [
        ['001', '051', '101', '151', '201', '251', '301', '351'],  # Group 1
        ['002', '052', '102', '152', '202', '252', '302', '352'],  # Group 2
        ['004', '054', '104', '154', '204', '254', '304', '354'],  # Group 3
        ['005', '055', '105', '155', '205', '255', '305', '355'],  # Group 4
        ['011', '061', '111', '161', '211', '261', '311', '361'],  # Group 5
        ['012', '062', '112', '162', '212', '262', '312', '362'],  # Group 6
        ['018', '068', '118', '168', '218', '268', '318', '368'],  # Group 7
        ['019', '069', '119', '169', '219', '269', '319', '369'],  # Group 8
        ['022', '072', '122', '172', '222', '272', '322', '372'],  # Group 9
        ['025', '075', '125', '175', '225', '275', '325', '375']   # Group 10
    ]

    # Prepare data to write to Excel
    detailed_results = []  # 存储所有详细结果
    group_results = []    # 存储每组平均结果

    # Process each patient
    for patient_id in patient_ids:
        full_dose_filename = f'P{patient_id}-corrsta-36size.tif'
        gen_dose_filename = f'P{patient_id}_pred.tif'
        full_dose_file = os.path.join(full_dose_folder, full_dose_filename)
        gen_dose_file = os.path.join(gen_dose_folder, gen_dose_filename)

        # Check if files exist
        if not os.path.exists(full_dose_file):
            print(f"Warning: Full-dose file not found for {full_dose_filename}")
            continue
        if not os.path.exists(gen_dose_file):
            print(f"Warning: Generated dose file not found for {gen_dose_filename}")
            continue

        # Read images
        full_dose_img = imread(full_dose_file)
        gen_dose_img = imread(gen_dose_file)

        # Verify image shapes
        if full_dose_img.shape != (36, 36, 36):
            print(f"Skipping {full_dose_filename}: Expected shape (36, 36, 36), got {full_dose_img.shape}")
            continue
        if gen_dose_img.shape != (36, 36, 36):
            print(f"Skipping {gen_dose_filename}: Expected shape (36, 36, 36), got {gen_dose_img.shape}")
            continue

        print(f"Processing pair: {full_dose_filename} vs {gen_dose_filename}")

        # Normalize images for PSNR, SSIM, NMSE
        full_dose_img_norm = normalize_image(full_dose_img)
        gen_dose_img_norm = normalize_image(gen_dose_img)

        # Calculate metrics
        h = np.size(full_dose_img)

        # ME and MAE
        ME = np.sum(gen_dose_img - full_dose_img) / h
        MAE = np.abs(ME)

        # NMSE (using normalized images)
        NMSE = mean_squared_error(full_dose_img_norm.flatten(),
                                 gen_dose_img_norm.flatten()) / np.mean(full_dose_img_norm ** 2)

        # PSNR
        MSE = mean_squared_error(full_dose_img_norm.flatten(), gen_dose_img_norm.flatten())
        PSNR = 10 * np.log10(1.0 / MSE) if MSE != 0 else float('inf')

        # SSIM
        SSIM = structural_similarity(full_dose_img_norm, gen_dose_img_norm,
                                    data_range=1.0, multichannel=False)

        # Store detailed results
        detailed_results.append([full_dose_filename, gen_dose_filename, ME, MAE, NMSE, PSNR, SSIM])

    # Calculate group averages
    for group_idx, group_ids in enumerate(groups, 1):
        group_metrics = {'ME': [], 'MAE': [], 'NMSE': [], 'PSNR': [], 'SSIM': []}
        for patient_id in group_ids:
            # Find metrics for this patient
            for result in detailed_results:
                if f'P{patient_id}-corrsta-36size.tif' in result[0]:
                    group_metrics['ME'].append(result[2])
                    group_metrics['MAE'].append(result[3])
                    group_metrics['NMSE'].append(result[4])
                    group_metrics['PSNR'].append(result[5])
                    group_metrics['SSIM'].append(result[6])

        # Calculate averages if group has data
        if group_metrics['ME']:
            group_results.append([
                f'Group {group_idx}',
                ','.join([f'P{id}' for id in group_ids]),
                np.mean(group_metrics['ME']),
                np.mean(group_metrics['MAE']),
                np.mean(group_metrics['NMSE']),
                np.mean(group_metrics['PSNR']),
                np.mean(group_metrics['SSIM'])
            ])

    # Save results to Excel
    detailed_df = pd.DataFrame(detailed_results,
                              columns=['Full_Dose_Filename', 'Gen_Dose_Filename', 'ME', 'MAE', 'NMSE', 'PSNR', 'SSIM'])
    group_df = pd.DataFrame(group_results,
                           columns=['Group', 'Patient_IDs', 'Avg_ME', 'Avg_MAE', 'Avg_NMSE', 'Avg_PSNR', 'Avg_SSIM'])

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(QA_save_path, f'QA_results_36size_3d_{timestamp}.xlsx')

    # Write to Excel with two sheets
    with pd.ExcelWriter(result_file, engine='openpyxl') as writer:
        detailed_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
        group_df.to_excel(writer, sheet_name='Group_Average_Results', index=False)

    print(f"Results saved to {result_file}")


if __name__ == '__main__':
    # Define paths
    full_dose_folder = '/hy-tmp/8cgspect_ground'  # 全剂量图像文件夹
    gen_dose_folder = '../output/corediff_8cgspect_1000t/save_tif3d/epoch_150000'   # 降噪图像文件夹
    QA_save_path = '../output/corediff_8cgspect_1000t'        # 结果保存路径

    # Ensure result directory exists
    os.makedirs(QA_save_path, exist_ok=True)

    # Run processing
    data_show(full_dose_folder, gen_dose_folder, QA_save_path)

    # 验证图像范围（可选）
    # import tifffile
    # full_dose_img = tifffile.imread(os.path.join(full_dose_folder, 'P001-corrsta-36size.tif'))
    # print(f"Full dose range: {full_dose_img.min()} to {full_dose_img.max()}")