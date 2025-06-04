import tqdm
from utils.measure import compute_measure  # 导入计算指标的函数
import wandb
import os.path as osp
from glob import glob
import numpy as np
import torch
import torchvision
from utils.loggerx import LoggerX
from corediff.corediff_wrapper import Network, WeightNet
from corediff.diffusion_modules import Diffusion
import os
from utils.ima_tif import process_tif_files
import pydicom  # Import for handling DICOM/IMA data
class SpectDataLoader:
    def __init__(self, data_root, patient_ids):
        self.data_root = data_root
        self.patient_ids = patient_ids
        self.base_input, self.base_target = self.process_data()

    def process_data(self):
        base_input, base_target = [], []

        # 目标图像路径
        for id in self.patient_ids:
            target_paths = sorted(glob(osp.join(self.data_root, f'P{id}_target_*_img.npy')))
            base_target += target_paths[1:len(target_paths) - 1]  # 去掉首尾文件

        dose = 10
        # 输入图像路径
        for id in self.patient_ids:
            input_paths = sorted(glob(osp.join(self.data_root, f'P{id}_{dose}_*_img.npy')))
            cat_patient_list = []
            for i in range(1, len(input_paths) - 1):
                patient_path = ''
                for j in range(-1, 2):
                    patient_path = patient_path + '~' + input_paths[i + j]
                cat_patient_list.append(patient_path)
            base_input = base_input + cat_patient_list

        return base_input, base_target

    def __getitem__(self, index):
        input, target = self.base_input[index], self.base_target[index]

        input_imgs = [np.load(img_path)[np.newaxis, ...].astype(np.float32) for img_path in input.split('~')[1:]]
        input = np.concatenate(input_imgs, axis=0)  # 3通道

        target = np.load(target)[np.newaxis, ...].astype(np.float32)  # 目标数据
        
        NMSE_scale = np.sum(target) / np.sum(input)
        input = input * NMSE_scale
        
        input, target = self.normalize_(input), self.normalize_(target)

        return input, target

    def __len__(self):
        return len(self.base_target)

    # def normalize_(self, img, MIN_B=0, MAX_B=65535):
    #     img[img < MIN_B] = MIN_B
    #     img[img > MAX_B] = MAX_B
    #     return (img - MIN_B) / (MAX_B - MIN_B)
    def normalize_(self, img):
        MIN_B = np.min(img)  # 动态计算图像的最小值
        MAX_B = np.max(img)  # 动态计算图像的最大值
        img[img < MIN_B] = MIN_B
        img[img > MAX_B] = MAX_B
        return (img - MIN_B) / (MAX_B - MIN_B)  # 归一化到 [0, 1]


class Model:
    def __init__(self, test_loader, ema_model, logger):
        self.test_loader = test_loader
        self.ema_model = ema_model
        self.logger = logger
        self.T = 5  # 采样步数
        self.sampling_routine = "ddim"  # 使用ddim采样
        self.start_adjust_iter = 1  # 采样调整起始步数
        self.dose = 10
        self.context = True

    @torch.no_grad()
    def test(self):
        best_model_path = osp.join(self.logger.models_save_dir, 'ema_model-best')
        self.ema_model.load_state_dict(torch.load(best_model_path))
        print("Loaded best model for testing.")
        self.ema_model.eval()
        n_iter = 150000  # 测试迭代次数
        psnr, ssim, nmse = 0., 0., 0.

        for low_dose, full_dose in tqdm.tqdm(self.test_loader, desc='test'):
            low_dose, full_dose = low_dose.cuda(), full_dose.cuda()

            gen_full_dose, _, _ = self.ema_model.sample(
                batch_size=low_dose.shape[0],
                img=low_dose,
                t=self.T,
                sampling_routine=self.sampling_routine,
                start_adjust_iter=self.start_adjust_iter,
            )

            # 计算PSNR、SSIM和NMSE
            # **最大值匹配**：将去噪图像的最大值调整到与全剂量图像相同的最大值
            Denoised_ary = gen_full_dose.cpu().numpy()  # 将 tensor 转换为 numpy
            Fulldose_ary = full_dose.cpu().numpy()  # 同样处理 full_dose
            max_full_dose = np.max(Fulldose_ary)
            max_denoised = np.max(Denoised_ary)
            Denoised_ary = Denoised_ary * (max_full_dose / max_denoised)
            psnr_score, ssim_score, nmse_score = compute_measure(full_dose, Denoised_ary, data_range=max_full_dose)
            psnr += psnr_score / len(self.test_loader)
            ssim += ssim_score / len(self.test_loader)
            nmse += nmse_score / len(self.test_loader)

        self.logger.msg([psnr, ssim, nmse], n_iter)
        self.logger.record_metrics(n_iter, psnr, ssim, nmse)

    def transfer_calculate_window(self, img, MIN_B=0, MAX_B=65535, cut_min=0, cut_max=65535):
        img = img * (MAX_B - MIN_B) + MIN_B  # 将归一化图像恢复到物理值范围
        img[img < cut_min] = cut_min  # 裁剪低于 cut_min 的像素
        img[img > cut_max] = cut_max  # 裁剪高于 cut_max 的像素
        img = 255 * (img - cut_min) / (cut_max - cut_min)  # 映射到 [0, 255] 用于显示
        return img
    # def transfer_display_window(self, img, MIN_B=0, MAX_B=65535, cut_min=0, cut_max=65535):
    #     img = img * (MAX_B - MIN_B) + MIN_B  # 将归一化图像恢复到物理值范围 [MIN_B, MAX_B]
    #     img[img < cut_min] = cut_min  # 将低于 cut_min 的像素值裁剪为 cut_min
    #     img[img > cut_max] = cut_max  # 将高于 cut_max 的像素值裁剪为 cut_max
    #     # 将图像值重新映射到 [0, 1] 的范围，用于显示或进一步处理
    #     img = (img - cut_min) / (cut_max - cut_min)
    #     return img  # 返回去归一化并且适合显示的图像
    def transfer_display_window(self, img, cut_min=None, cut_max=None):
        # 如果没有提供 cut_min 和 cut_max，使用图像的最小值和最大值
        if cut_min is None:
            cut_min = np.min(img)  # 动态计算图像最小值
        if cut_max is None:
            cut_max = np.max(img)  # 动态计算图像最大值

        # 归一化到 [0, 1] 范围
        img = (img - cut_min) / (cut_max - cut_min)

        # 将图像值映射到 [0, 255] 用于显示
        img = 255 * img
        return img

    @torch.no_grad()  # 禁用梯度计算，提高推理效率，减少内存消耗
    def generate_images(self):
        best_model_path = osp.join(self.logger.models_save_dir, 'ema_model-best')
        self.ema_model.load_state_dict(torch.load(best_model_path))
        print("Loaded best model for testing.")
        self.ema_model.eval()
        n_iter = 150000  # 测试迭代次数
        
        low_dose, full_dose = forgen_test_images  # 从预定义的测试图像中获取低剂量和全剂量图像

        # 使用 EMA 模型生成预测的全剂量图像以及中间的重建图像
        gen_full_dose, direct_recons, imstep_imgs = self.ema_model.sample(
            batch_size=low_dose.shape[0],  # 设定 batch 大小
            img=low_dose,  # 输入低剂量图像
            t=self.T,  # 设定扩散过程的时间步数
            sampling_routine=self.sampling_routine,  # 设定采样例程
            n_iter=n_iter,  # 当前迭代步数
            start_adjust_iter=self.start_adjust_iter,  # 调整开始的迭代步数
        )

        low1_dose = low_dose[:, 1].unsqueeze(1)
        b, c, w, h = low1_dose.size()  # 获取图像的 batch 大小、通道数、宽度和高度
        # 遍历每个图像，并根据其文件名提取文件ID和断层数
        for i in range(b):
            # 从输入路径提取文件 ID 和断层号
            input_file_path = dataset.base_input[i]

            context_frames = input_file_path.split('~')  # 切割上下文帧路径
            first_frame_path = context_frames[1]  # 获取第一帧 (索引 1 是第一个有效路径)
            base_filename = os.path.basename(first_frame_path)  # 获取第一帧的文件名
       
            # 文件名示例：'P01_10_000_img.npy'
            file_id = base_filename.split('_')[0]  # 提取文件ID，例如 'P01'
            slice_num = base_filename.split('_')[2]  # 提取断层数，例如 '000'

            # 将低剂量图像、全剂量图像和生成的全剂量图像堆叠到一个张量中
            fake_imgs = torch.stack([low1_dose[i], full_dose[i], gen_full_dose[i]])
            fake_imgs = self.transfer_display_window(fake_imgs)
            fake_imgs = fake_imgs.transpose(1, 0).reshape((-1, c, w, h))

            # 保存图像，文件名带上 file_id 和 slice_num (基于第一帧文件名)
            self.logger.save_image(
                torchvision.utils.make_grid(fake_imgs, nrow=3),
                n_iter,
                f'test_{self.dose}_{self.sampling_routine}_{file_id}_{slice_num}'
            )
            self.logger.save_tif(gen_full_dose[i], n_iter,
                                 f'test_{self.dose}_{self.sampling_routine}_{file_id}_{slice_num}')


# 初始化数据和模型
data_root = '../data_preprocess/gen_data/spect_10s1s_npy'
patient_ids = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
dataset = SpectDataLoader(data_root, patient_ids)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

# 加载全部测试图像用于测试和可视化
test_images = [dataset[i] for i in range(len(dataset))]
low_dose = torch.stack([torch.from_numpy(x[0]) for x in test_images], dim=0).cuda()
full_dose = torch.stack([torch.from_numpy(x[1]) for x in test_images], dim=0).cuda()
# 保存低剂量和全剂量图像
forgen_test_images = (low_dose, full_dose)
# 保存测试数据集
# self.test_dataset = test_dataset

denoise_fn = Network(in_channels=3, context=True)
# 初始化模型和日志记录器
ema_model = Diffusion(
            denoise_fn=denoise_fn,
            image_size=64,
            timesteps=5,
            context=True
        ).cuda()
logger = LoggerX(save_root=osp.join(
    osp.dirname(osp.dirname(osp.abspath(__file__))), 'output', 'corediff_dose10s1s_spect_5t'
))

# 运行测试
model = Model(test_loader, ema_model, logger)
model.test()
model.generate_images()
# 示例：处理输入文件夹中的 IMA 文件并保存为 TIF 格式
input_folder = "../output/corediff_dose10s1s_spect_5t/save_IMA"  # IMA 文件所在文件夹
output_folder = "../output/corediff_dose10s1s_spect_5t/save_tif"  # 输出的 TIF 文件夹
process_tif_files(input_folder, output_folder)