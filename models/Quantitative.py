import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
import xlwt
from tifffile import imread

def data_show(file_path, QA_save_path):
    ME1_list = []
    MAE1_list =[]
    RMSE1_list = []
    psnr1_list = []
    ssim1_list = []

    ME2_list = []
    MAE2_list =[]
    RMSE2_list = []
    psnr2_list = []
    ssim2_list = []

    workbook = xlwt.Workbook()

    sheet=workbook.add_sheet('result', cell_overwrite_ok=True)

    for path, subdirs, files in os.walk(file_path):
        for i in range (len(files)):
            Fulldose_name = path + files[i]
            Denoised_name = Fulldose_name.replace('Fulldose\\','Denoised\\')
            Denoised_name = Denoised_name.replace('10s_06gauss_cVOI.tif', '1s_AttcGANprj_4defsli_cVOI.tif')
            Fulldose_nii = imread(Fulldose_name)
            Denoised_nii = imread(Denoised_name)
            Denoised_ary = Denoised_nii
            Fulldose_ary = Fulldose_nii
            h = np.size(Fulldose_ary)
            
            NMSE_scale = np.sum(Fulldose_ary) / np.sum(Denoised_ary)
            Denoised_ary = Denoised_ary * NMSE_scale

            ME= np.sum(Denoised_ary - Fulldose_ary)/h
            MAE = np.abs(ME)
            NMSE = mean_squared_error(Fulldose_ary, Denoised_ary)/np.mean(Fulldose_ary**2)
            PSNR = peak_signal_noise_ratio(Fulldose_ary, Denoised_ary, data_range=np.max(Fulldose_ary))
            ssim = structural_similarity(Fulldose_ary, Denoised_ary, data_range=np.max(Fulldose_ary))

            sheet.write(i, 0, str(files[i][0:6]))
            sheet.write(i, 1, str(ME))
            sheet.write(i, 2, str(MAE))
            sheet.write(i, 3, str(NMSE))
            sheet.write(i, 4, str(PSNR))
            sheet.write(i, 5, str(ssim))

    workbook.save(QA_save_path+'1s_AttcGANprj_4defsli_cVOI2023'+'.xls')

if __name__ == '__main__':
    file_path = 'F:\\1.Paper3\\Data analysis\\Fulldose\\'
    QA_save_path = 'F:\\1.Paper3\\Data analysis\\Result\\'
    if not os.path.exists(QA_save_path):
        os.mkdir(QA_save_path)
    data_show(file_path, QA_save_path)




