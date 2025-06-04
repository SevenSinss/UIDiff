import os
from skimage.io import imread
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from sklearn.metrics import mean_squared_error
import numpy as np
import xlwt
from datetime import datetime
from skimage.exposure import match_histograms


def match_histograms_(input_img, target_img):
    """
    Match histograms of input to target.

    Args:
        input_img: Input image array (e.g., shape [C, H, W]).
        target_img: Target image array (e.g., shape [C, H, W] or [1, H, W]).

    Returns:
        Matched input image with the same shape as input_img.
    """
    # Ensure target image has the same number of channels
    if target_img.shape[0] != input_img.shape[0]:
        # If target has only one channel, repeat it to match input channels
        target_img = np.repeat(target_img, input_img.shape[0], axis=0)

    # Perform histogram matching channel by channel
    matched_channels = []
    for c in range(input_img.shape[0]):
        matched_channel = match_histograms(
            input_img[c], target_img[c], channel_axis=None
        )
        matched_channels.append(matched_channel)

    # Stack channels back together
    matched_img = np.stack(matched_channels, axis=0)
    return matched_img.astype(np.float32)

# def data_show(full_dose_path, denoised_path, QA_save_path):
#     # Initialize Excel Workbook
#     workbook = xlwt.Workbook()
#     sheet = workbook.add_sheet('result', cell_overwrite_ok=True)
#
#     # Write Excel Headers
#     headers = ['Filename', 'ME', 'MAE', 'NMSE', 'PSNR', 'SSIM']
#     for col, header in enumerate(headers):
#         sheet.write(0, col, header)
#
#     # Get sorted file lists
#     full_dose_files = sorted(os.listdir(full_dose_path))
#     denoised_files = sorted(os.listdir(denoised_path))
#
#     if len(full_dose_files) == 0 or len(denoised_files) == 0:
#         raise ValueError("One or both directories are empty. Please check the input paths.")
#
#     # Process files
#     for i, filename in enumerate(full_dose_files):
#         full_dose_file = os.path.join(full_dose_path, filename)
#         denoised_file = os.path.join(denoised_path, filename)
#
#         if not os.path.exists(denoised_file):
#             print(f"Warning: Denoised file not found for {filename}")
#             continue
#
#         # Read Images
#         full_dose_img = imread(full_dose_file)
#         denoised_img = imread(denoised_file)
#
#         # Ensure images have the same dimensions
#         if full_dose_img.shape != denoised_img.shape:
#             print(f"Skipping {filename}: Dimension mismatch between full dose and denoised images.")
#             continue
#
#         print(f"Processing file: {filename}")
#         print(f"Full dose image - min: {full_dose_img.min()}, max: {full_dose_img.max()}")
#         print(f"Denoised image - min: {denoised_img.min()}, max: {denoised_img.max()}")
#
#         # Adjust intensity of denoised image to match full dose image
#         denoised_img_scaled = denoised_img * (np.sum(full_dose_img) / np.sum(denoised_img))
#         print(f"Denoised image_scaled - min: {denoised_img_scaled.min()}, max: {denoised_img_scaled.max()}")
#
#         # Normalize images to range [0, 1]
#         full_dose_img_normalized = full_dose_img / np.max(full_dose_img)
#         denoised_img_scaled_normalized = denoised_img_scaled / np.max(denoised_img_scaled)
#
#         denoised_img_scaled_normalized = match_histograms_(denoised_img_scaled_normalized, full_dose_img_normalized)
#         # Flatten the images to 1D arrays for metric calculation
#         full_dose_img_flat = full_dose_img_normalized.ravel()
#         denoised_img_scaled_flat = denoised_img_scaled_normalized.ravel()
#
#         # Calculate Metrics
#         h = full_dose_img_flat.size  # Number of elements
#         ME = np.sum(denoised_img_scaled_flat - full_dose_img_flat) / h
#         MAE = np.abs(ME)
#         NMSE = mean_squared_error(full_dose_img_flat, denoised_img_scaled_flat) / np.mean(full_dose_img_flat ** 2)
#         err = mean_squared_error(full_dose_img_flat, denoised_img_scaled_flat)
#
#         if err == 0:
#             PSNR = float('inf')  # PSNR is theoretically infinite
#         else:
#             PSNR = peak_signal_noise_ratio(full_dose_img_flat, denoised_img_scaled_flat, data_range=1.0)
#
#         # Check if PSNR is invalid
#         if np.isnan(PSNR) or np.isinf(PSNR):
#             PSNR = 100  # Set a reasonable default value if PSNR is invalid
#
#         SSIM = structural_similarity(full_dose_img_flat, denoised_img_scaled_flat, data_range=1.0)
#
#         # Write Results to Excel
#         sheet.write(i + 1, 0, filename)
#         sheet.write(i + 1, 1, ME)
#         sheet.write(i + 1, 2, MAE)
#         sheet.write(i + 1, 3, NMSE)
#         sheet.write(i + 1, 4, PSNR)
#         sheet.write(i + 1, 5, SSIM)
#
#     # Save Excel File
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     result_file = os.path.join(QA_save_path, f'QA_results_{timestamp}.xls')
#     workbook.save(result_file)
#     print(f"Results saved to {result_file}")

# def data_show(full_dose_path, denoised_path, QA_save_path):
#     # Initialize Excel Workbook
#     workbook = xlwt.Workbook()
#     sheet = workbook.add_sheet('result', cell_overwrite_ok=True)
#
#     # Write Excel Headers
#     headers = ['Filename', 'ME', 'MAE', 'NMSE', 'PSNR', 'SSIM']
#     for col, header in enumerate(headers):
#         sheet.write(0, col, header)
#
#     # Get sorted file lists for denoised (1s) and full dose (10s)
#     denoised_files = sorted([f for f in os.listdir(denoised_path) if f.endswith('_2s.tif')])
#     full_dose_files = sorted([f for f in os.listdir(full_dose_path) if f.endswith('_10s.tif')])
#
#     if len(denoised_files) == 0 or len(full_dose_files) == 0:
#         raise ValueError("One or both directories are empty. Please check the input paths.")
#
#     # Process files
#     for i, denoised_filename in enumerate(denoised_files):
#         # Extract the patient ID from the denoised file name (e.g., data1, data2, ...)
#         patient_id = denoised_filename.split('_')[0]  # Assumes filename starts with patient ID like 'data1'
#         full_dose_filename = f"{patient_id}_ld_recon_clini_10s.tif"  # Corresponding full dose file
#
#         # Check if the corresponding full dose file exists
#         if full_dose_filename not in full_dose_files:
#             print(f"Warning: Full dose file not found for {denoised_filename}")
#             continue
#
#         # Read Images
#         full_dose_file = os.path.join(full_dose_path, full_dose_filename)
#         denoised_file = os.path.join(denoised_path, denoised_filename)
#
#         # Read Images
#         full_dose_img = imread(full_dose_file)
#         denoised_img = imread(denoised_file)
#
#         # Ensure images have the same dimensions
#         if full_dose_img.shape != denoised_img.shape:
#             print(f"Skipping {denoised_filename}: Dimension mismatch between full dose and denoised images.")
#             continue
#
#         print(f"Processing file: {denoised_filename}")
#         print(f"Full dose image - min: {full_dose_img.min()}, max: {full_dose_img.max()}")
#         print(f"Denoised image - min: {denoised_img.min()}, max: {denoised_img.max()}")
#
#         # Adjust intensity of denoised image to match full dose image
#         denoised_img_scaled = denoised_img * (np.sum(full_dose_img) / np.sum(denoised_img))
#         print(f"Denoised image_scaled - min: {denoised_img_scaled.min()}, max: {denoised_img_scaled.max()}")
#
#         # Normalize images to range [0, 1]
#         full_dose_img_normalized = full_dose_img / np.max(full_dose_img)
#         denoised_img_scaled_normalized = denoised_img_scaled / np.max(denoised_img_scaled)
#
#         denoised_img_scaled_normalized = match_histograms_(denoised_img_scaled_normalized, full_dose_img_normalized)
#         # Flatten the images to 1D arrays for metric calculation
#         full_dose_img_flat = full_dose_img_normalized.ravel()
#         denoised_img_scaled_flat = denoised_img_scaled_normalized.ravel()
#
#         # Calculate Metrics
#         h = full_dose_img_flat.size  # Number of elements
#         ME = np.sum(denoised_img_scaled_flat - full_dose_img_flat) / h
#         MAE = np.abs(ME)
#         NMSE = mean_squared_error(full_dose_img_flat, denoised_img_scaled_flat) / np.mean(full_dose_img_flat ** 2)
#         err = mean_squared_error(full_dose_img_flat, denoised_img_scaled_flat)
#
#         if err == 0:
#             PSNR = float('inf')  # PSNR is theoretically infinite
#         else:
#             PSNR = peak_signal_noise_ratio(full_dose_img_flat, denoised_img_scaled_flat, data_range=1.0)
#
#         # Check if PSNR is invalid
#         if np.isnan(PSNR) or np.isinf(PSNR):
#             PSNR = 100  # Set a reasonable default value if PSNR is invalid
#
#         SSIM = structural_similarity(full_dose_img_flat, denoised_img_scaled_flat, data_range=1.0)
#
#         # Write Results to Excel
#         sheet.write(i + 1, 0, denoised_filename)
#         sheet.write(i + 1, 1, ME)
#         sheet.write(i + 1, 2, MAE)
#         sheet.write(i + 1, 3, NMSE)
#         sheet.write(i + 1, 4, PSNR)
#         sheet.write(i + 1, 5, SSIM)
#
#         # Save Excel File
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     result_file = os.path.join(QA_save_path, f'QA_results_{timestamp}.xls')
#     workbook.save(result_file)
#     print(f"Results saved to {result_file}")

# import pandas as pd
#
# def data_show(full_dose_path, denoised_path, QA_save_path):
#     # Prepare data to write to Excel
#     results = []
#
#     # Get sorted file lists
#     full_dose_files = sorted(os.listdir(full_dose_path))
#     denoised_files = sorted(os.listdir(denoised_path))
#
#     if len(full_dose_files) == 0 or len(denoised_files) == 0:
#         raise ValueError("One or both directories are empty. Please check the input paths.")
#
#     # Process files
#     for i, filename in enumerate(full_dose_files):
#         full_dose_file = os.path.join(full_dose_path, filename)
#         denoised_file = os.path.join(denoised_path, filename)
#
#         if not os.path.exists(denoised_file):
#             print(f"Warning: Denoised file not found for {filename}")
#             continue
#
#         # Read Images
#         full_dose_img = imread(full_dose_file)
#         denoised_img = imread(denoised_file)
#
#         # Ensure images have the same dimensions
#         if full_dose_img.shape != denoised_img.shape:
#             print(f"Skipping {filename}: Dimension mismatch between full dose and denoised images.")
#             continue
#
#         print(f"Processing file: {filename}")
#         print(f"Full dose image - min: {full_dose_img.min()}, max: {full_dose_img.max()}")
#         print(f"Denoised image - min: {denoised_img.min()}, max: {denoised_img.max()}")
#
#         Denoised_ary = denoised_img
#         Fulldose_ary = full_dose_img
#         print(Fulldose_ary.shape)
#         print(Denoised_ary.shape)
#
#         h = np.size(Fulldose_ary)
#
#         NMSE_scale = np.sum(Fulldose_ary) / np.sum(Denoised_ary)
#         Denoised_ary = Denoised_ary * NMSE_scale
#         degui = denoised_img.max() - denoised_img.min()
#         fdgui = full_dose_img.max() - full_dose_img.min()
#         print(f"Denoised max - min: {degui}, full max - min: {fdgui}")
#         Denoised_ary = Denoised_ary / degui
#         Fulldose_ary = Fulldose_ary / fdgui
#         # Denoised_ary = (Denoised_ary - np.min(Denoised_ary)) / (np.max(Denoised_ary) - np.min(Denoised_ary))
#         # Fulldose_ary = (Fulldose_ary - np.min(Fulldose_ary)) / (np.max(Fulldose_ary) - np.min(Fulldose_ary))
#
#         ME = np.sum(Denoised_ary - Fulldose_ary) / h
#         MAE = np.abs(ME)
#
#         NMSE_values = []
#         for i in range(Fulldose_ary.shape[0]):
#             NMSE = mean_squared_error(Fulldose_ary[i].flatten(), Denoised_ary[i].flatten()) / np.mean(
#                 Fulldose_ary[i] ** 2)
#             NMSE_values.append(NMSE)
#         average_NMSE = np.mean(NMSE_values)
#
#         # 直接对整张图像计算 NMSE
#         # average_NMSE = mean_squared_error(Fulldose_ary.flatten(), Denoised_ary.flatten()) / np.mean(Fulldose_ary ** 2)
#
#         # PSNR = peak_signal_noise_ratio(Fulldose_ary, Denoised_ary, data_range=np.max(Fulldose_ary))
#         MSE = average_NMSE * np.mean(Fulldose_ary ** 2)  # 从 NMSE 转换为 MSE
#         # 计算 PSNR
#         MAX_value = np.max(Fulldose_ary)  # 图像的最大值（例如归一化后的值为 1）
#         PSNR = 10 * np.log10((MAX_value ** 2) / MSE)
#
#         SSIM = structural_similarity(Fulldose_ary, Denoised_ary, data_range=np.max(Fulldose_ary))
#
#         print(
#             f"Writing to Excel: {filename}, ME: {ME}, MAE: {MAE}, NMSE: {average_NMSE}, PSNR: {PSNR}, SSIM: {SSIM}")
#
#         results.append([filename, ME, MAE, average_NMSE, PSNR, SSIM])
#
#     # Save results to Excel using pandas
#     df = pd.DataFrame(results, columns=['Filename', 'ME', 'MAE', 'NMSE', 'PSNR', 'SSIM'])
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     result_file = os.path.join(QA_save_path, f'QA_results_{timestamp}.xlsx')
#     df.to_excel(result_file, index=False)
#     print(f"Results saved to {result_file}")

# def data_show(full_dose_path, denoised_path, QA_save_path):
#     # Prepare data to write to Excel
#     results = []
#
#     # Get sorted file lists for denoised (1s) and full dose (10s)
#     denoised_files = sorted([f for f in os.listdir(denoised_path) if f.endswith('_5s.tif')])
#     full_dose_files = sorted([f for f in os.listdir(full_dose_path) if f.endswith('_10s.tif')])
#
#     if len(denoised_files) == 0 or len(full_dose_files) == 0:
#         raise ValueError("One or both directories are empty. Please check the input paths.")
#
#     # Process files
#     for i, denoised_filename in enumerate(denoised_files):
#         # Extract the patient ID from the denoised file name (e.g., data1, data2, ...)
#         patient_id = denoised_filename.split('_')[0]  # Assumes filename starts with patient ID like 'data1'
#         full_dose_filename = f"{patient_id}_ld_recon_clini_10s.tif"  # Corresponding full dose file
#
#         # Check if the corresponding full dose file exists
#         if full_dose_filename not in full_dose_files:
#             print(f"Warning: Full dose file not found for {denoised_filename}")
#             continue
#
#         # Read Images
#         full_dose_file = os.path.join(full_dose_path, full_dose_filename)
#         denoised_file = os.path.join(denoised_path, denoised_filename)
#
#         # Read Images
#         full_dose_img = imread(full_dose_file)
#         denoised_img = imread(denoised_file)
#
#         # Ensure images have the same dimensions
#         if full_dose_img.shape != denoised_img.shape:
#             print(f"Skipping {denoised_filename}: Dimension mismatch between full dose and denoised images.")
#             continue
#
#         print(f"Processing file: {denoised_filename}")
#         print(f"Full dose image - min: {full_dose_img.min()}, max: {full_dose_img.max()}")
#         print(f"Denoised image - min: {denoised_img.min()}, max: {denoised_img.max()}")
#
#         Denoised_ary = denoised_img
#         Fulldose_ary = full_dose_img
#         print(Fulldose_ary.shape)
#         print(Denoised_ary.shape)
#
#         h = np.size(Fulldose_ary)
#
#         NMSE_scale = np.sum(Fulldose_ary) / np.sum(Denoised_ary)
#         Denoised_ary = Denoised_ary * NMSE_scale
#         degui = denoised_img.max() - denoised_img.min()
#         fdgui = full_dose_img.max() - full_dose_img.min()
#         print(f"Denoised max - min: {degui}, full max - min: {fdgui}")
#         Denoised_ary = Denoised_ary / degui
#         Fulldose_ary = Fulldose_ary / fdgui
#
#         ME = np.sum(Denoised_ary - Fulldose_ary) / h
#         MAE = np.abs(ME)
#
#         NMSE_values = []
#         for i in range(Fulldose_ary.shape[0]):
#             NMSE = mean_squared_error(Fulldose_ary[i].flatten(), Denoised_ary[i].flatten()) / np.mean(
#                 Fulldose_ary[i] ** 2)
#             NMSE_values.append(NMSE)
#         average_NMSE = np.mean(NMSE_values)
#
#         # 直接对整张图像计算 NMSE
#         # average_NMSE = mean_squared_error(Fulldose_ary.flatten(), Denoised_ary.flatten()) / np.mean(Fulldose_ary ** 2)
#
#         # PSNR = peak_signal_noise_ratio(Fulldose_ary, Denoised_ary, data_range=np.max(Fulldose_ary))
#         MSE = average_NMSE * np.mean(Fulldose_ary ** 2)  # 从 NMSE 转换为 MSE
#         # 计算 PSNR
#         MAX_value = np.max(Fulldose_ary)  # 图像的最大值（例如归一化后的值为 1）
#         PSNR = 10 * np.log10((MAX_value ** 2) / MSE)
#
#         SSIM = structural_similarity(Fulldose_ary, Denoised_ary, data_range=np.max(Fulldose_ary))
#
#         print(f"Writing to Excel: {denoised_filename}, ME: {ME}, MAE: {MAE}, NMSE: {average_NMSE}, PSNR: {PSNR}, SSIM: {SSIM}")
#
#         results.append([denoised_filename, ME, MAE, average_NMSE, PSNR, SSIM])
#
#     # Save results to Excel using pandas
#     df = pd.DataFrame(results, columns=['Filename', 'ME', 'MAE', 'NMSE', 'PSNR', 'SSIM'])
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     result_file = os.path.join(QA_save_path, f'QA_results_{timestamp}.xlsx')
#     df.to_excel(result_file, index=False)
#     print(f"Results saved to {result_file}")


#True
import os
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from datetime import datetime
from tifffile import imread


def data_show(full_dose_path, denoised_path, QA_save_path):
    # Prepare data to write to Excel
    results = []

    # Get sorted file lists
    full_dose_files = sorted(os.listdir(full_dose_path))
    denoised_files = sorted(os.listdir(denoised_path))

    if len(full_dose_files) == 0 or len(denoised_files) == 0:
        raise ValueError("One or both directories are empty. Please check the input paths.")

    # Process files
    for i, filename in enumerate(full_dose_files):
        full_dose_file = os.path.join(full_dose_path, filename)
        denoised_file = os.path.join(denoised_path, filename)

        if not os.path.exists(denoised_file):
            print(f"Warning: Denoised file not found for {filename}")
            continue

        # Read Images
        full_dose_img = imread(full_dose_file)
        denoised_img = imread(denoised_file)

        # Ensure images have the same dimensions
        if full_dose_img.shape != denoised_img.shape:
            print(f"Skipping {filename}: Dimension mismatch between full dose and denoised images.")
            continue

        print(f"Processing file: {filename}")

        Denoised_ary = denoised_img
        Fulldose_ary = full_dose_img
        h = np.size(Fulldose_ary)

        # **最大值匹配**：将去噪图像的最大值调整到与全剂量图像相同的最大值
        max_full_dose = np.max(Fulldose_ary)
        max_denoised = np.max(Denoised_ary)
        Denoised_ary = Denoised_ary * (max_full_dose / max_denoised)

        # ME and MAE Calculation
        ME = np.sum(Denoised_ary - Fulldose_ary) / h
        MAE = np.abs(ME)

        # NMSE calculation (using per-slice calculation)
        NMSE_values = []
        for slice_idx in range(Fulldose_ary.shape[0]):  # Iterate over slices (assuming 3D data)
            NMSE = mean_squared_error(Fulldose_ary[slice_idx].flatten(), Denoised_ary[slice_idx].flatten()) / np.mean(
                Fulldose_ary[slice_idx] ** 2)
            NMSE_values.append(NMSE)
        average_NMSE = np.mean(NMSE_values)

        # PSNR calculation
        MSE = average_NMSE * np.mean(Fulldose_ary ** 2)
        PSNR = 10 * np.log10((max_full_dose ** 2) / MSE)

        # SSIM calculation
        SSIM = structural_similarity(Fulldose_ary, Denoised_ary, data_range=np.max(Fulldose_ary))

        print(f"Writing to Excel: {filename}, ME: {ME}, MAE: {MAE}, NMSE: {average_NMSE}, PSNR: {PSNR}, SSIM: {SSIM}")
        results.append([filename, ME, MAE, average_NMSE, PSNR, SSIM])

    # Save results to Excel
    df = pd.DataFrame(results, columns=['Filename', 'ME', 'MAE', 'NMSE', 'PSNR', 'SSIM'])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(QA_save_path, f'QA_results_{timestamp}.xlsx')
    df.to_excel(result_file, index=False)
    print(f"Results saved to {result_file}")

if __name__ == '__main__':
    # Define paths
    full_dose_path = r'C:\Users\Never\Desktop\spectdata32\group\g1\ground_10s'
    denoised_path = r'C:\Users\Never\Desktop\spectdata32\group\g1\ground_10s'
    QA_save_path = r'C:\Users\Never\Desktop\spectdata32\group\results\ground_1s'

    # full_dose_path = r'C:\Users\Never\Desktop\spdata\tif_10s'
    # denoised_path = r'C:\Users\Never\Desktop\spdata\tif_5s'
    # QA_save_path = r'C:\Users\Never\Desktop\spdata\AAresults\5s'

    # Ensure result directory exists
    os.makedirs(QA_save_path, exist_ok=True)

    # Run processing
    data_show(full_dose_path, denoised_path, QA_save_path)



