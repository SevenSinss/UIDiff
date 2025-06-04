import tqdm
from utils.measure import compute_measure
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
import pydicom
import re
import time


class SpectDataLoader:
    def __init__(self, data_root, patient_ids):
        self.data_root = data_root
        self.patient_ids = patient_ids
        self.base_input, self.base_target = self.process_data()

    def process_data(self):
        base_input, base_target = [], []

        for id in self.patient_ids:
            target_paths = sorted(glob(osp.join(self.data_root, f'P{id}_target_*_img.npy')))
            base_target += target_paths[1:len(target_paths) - 1]

        dose = '8cg'
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
        input = np.concatenate(input_imgs, axis=0)

        target = np.load(target)[np.newaxis, ...].astype(np.float32)

        NMSE_scale = np.sum(target) / np.sum(input)
        input = input * NMSE_scale

        input = self.normalize_(input)  # 仅归一化 input
        # target 不归一化，保留原始物理值范围

        return input, target

    def __len__(self):
        return len(self.base_target)

    def normalize_(self, img):
        MIN_B = np.min(img)
        MAX_B = np.max(img)
        if MAX_B <= MIN_B:
            return img  # 避免除零
        img[img < MIN_B] = MIN_B
        img[img > MAX_B] = MAX_B
        return (img - MIN_B) / (MAX_B - MIN_B)


class Model:
    def __init__(self, test_loader, ema_model, logger, test_images):
        self.test_loader = test_loader
        self.ema_model = ema_model
        self.logger = logger
        self.test_images = test_images
        self.T = 1000
        self.sampling_routine = "ddim"
        self.start_adjust_iter = 1
        self.dose = '8cg'
        self.context = True

    @torch.no_grad()
    def test(self):
        best_model_path = osp.join(self.logger.models_save_dir, 'ema_model-20000')
        try:
            self.ema_model.load_state_dict(torch.load(best_model_path))
            print("Loaded best model for testing.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file {best_model_path} not found.")
        self.ema_model.eval()
        n_iter = 150000
        psnr, ssim, nmse = 0., 0., 0.
        valid_batches = 0

        for low_dose, full_dose in tqdm.tqdm(self.test_loader, desc='test'):
            low_dose, full_dose = low_dose.cuda(), full_dose.cuda()
            gen_full_dose, _, _ = self.ema_model.sample(
                batch_size=low_dose.shape[0],
                img=low_dose,
                t=self.T,
                sampling_routine=self.sampling_routine,
                start_adjust_iter=self.start_adjust_iter,
            )
            # 归一化 full_dose 和 gen_full_dose 以计算指标
            full_dose_norm = self.normalize(full_dose)
            gen_full_dose_norm = self.normalize(gen_full_dose)
            data_range = np.max(full_dose_norm) - np.min(full_dose_norm)
            psnr_score, ssim_score, nmse_score = compute_measure(full_dose_norm, gen_full_dose_norm, data_range=data_range)
            psnr += psnr_score
            ssim += ssim_score
            nmse += nmse_score
            valid_batches += 1

        if valid_batches > 0:
            psnr /= valid_batches
            ssim /= valid_batches
            nmse /= valid_batches
        self.logger.msg([psnr, ssim, nmse], n_iter)
        self.logger.record_metrics(n_iter, psnr, ssim, nmse)

    def normalize(self, img):
        """辅助函数：归一化图像到 [0, 1]，始终返回 numpy.ndarray"""
        img_np = img.cpu().numpy() if isinstance(img, torch.Tensor) else img
        MIN_B = np.min(img_np)
        MAX_B = np.max(img_np)
        if MAX_B <= MIN_B:
            return img_np
        img_np = (img_np - MIN_B) / (MAX_B - MIN_B)
        return img_np

    def transfer_calculate_window(self, img, ref_img=None, for_display=True, cut_min=None, cut_max=None):
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if isinstance(ref_img, torch.Tensor):
            ref_img = ref_img.cpu().numpy()

        if ref_img is not None:
            MIN_B = np.min(ref_img)
            MAX_B = np.max(ref_img)
            if MAX_B <= MIN_B:
                print(f"Warning: Invalid ref_img range ({MIN_B:.6f}, {MAX_B:.6f}), using fallback [0, 1]")
                MIN_B = 0
                MAX_B = 1.0
        else:
            MIN_B = np.min(img)
            MAX_B = np.max(img)
            if MAX_B <= MIN_B:
                MIN_B = 0
                MAX_B = 1.0

        img = img * (MAX_B - MIN_B) + MIN_B

        if cut_min is None:
            cut_min = MIN_B
        if cut_max is None:
            cut_max = MAX_B
        img = np.clip(img, cut_min, cut_max)

        if for_display:
            if cut_max == cut_min:
                img = np.zeros_like(img)
            else:
                img = 255 * (img - cut_min) / (cut_max - cut_min)
            return img.astype(np.uint8)
        else:
            return img

    def transfer_display_window(self, img, ref_img=None, cut_min=None, cut_max=None):
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if isinstance(ref_img, torch.Tensor):
            ref_img = ref_img.cpu().numpy()

        if ref_img is not None:
            MIN_B = np.min(ref_img)
            MAX_B = np.max(ref_img)
            if MAX_B <= MIN_B:
                print(f"Warning: Invalid ref_img range ({MIN_B:.6f}, {MAX_B:.6f}), using fallback [0, 1]")
                MIN_B = 0
                MAX_B = 1.0
        else:
            MIN_B = np.min(img)
            MAX_B = np.max(img)
            if MAX_B <= MIN_B:
                MIN_B = 0
                MAX_B = 1.0

        img = img * (MAX_B - MIN_B) + MIN_B
        img = np.clip(img, cut_min if cut_min is not None else MIN_B, cut_max if cut_max is not None else MAX_B)

        if cut_max == cut_min:
            img = np.zeros_like(img)
        else:
            img = (img - cut_min if cut_min is not None else MIN_B) / (cut_max - cut_min if cut_max is not None else MAX_B - MIN_B)

        img = 255 * img
        return img

    @torch.no_grad()
    def generate_images(self):
        best_model_path = osp.join(self.logger.models_save_dir, 'ema_model-20000')
        try:
            self.ema_model.load_state_dict(torch.load(best_model_path))
            print("Loaded best model for testing.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file {best_model_path} not found.")
        self.ema_model.eval()
        n_iter = 150000

        low_dose, full_dose = self.test_images

        gen_full_dose, direct_recons, imstep_imgs = self.ema_model.sample(
            batch_size=low_dose.shape[0],
            img=low_dose,
            t=self.T,
            sampling_routine=self.sampling_routine,
            n_iter=n_iter,
            start_adjust_iter=self.start_adjust_iter,
        )

        print(f"gen_full_dose shape: {gen_full_dose.shape}, range: {gen_full_dose.min().item():.6f} to {gen_full_dose.max().item():.6f}")
        print(f"full_dose shape: {full_dose.shape}, range: {full_dose.min().item():.6f} to {full_dose.max().item():.6f}")

        low1_dose = low_dose[:, 1].unsqueeze(1)
        b, c, w, h = low1_dose.size()
        for i in range(b):
            input_file_path = self.test_loader.dataset.base_input[i]
            context_frames = input_file_path.split('~')
            first_frame_path = context_frames[1]
            base_filename = os.path.basename(first_frame_path)

            match = re.match(r'^(P\d+)_8cg_(\d+)_img\.npy$', base_filename)
            if not match:
                raise ValueError(f"Invalid filename format: {base_filename}")
            file_id, slice_num = match.groups()

            print(f"Image {i} ({file_id}_{slice_num}):")
            print(f"  gen_full_dose[{i}] range: {gen_full_dose[i:i+1].min().item():.6f} to {gen_full_dose[i:i+1].max().item():.6f}")
            print(f"  full_dose[{i}] range: {full_dose[i:i+1].min().item():.6f} to {full_dose[i:i+1].max().item():.6f}")

            fake_imgs = torch.stack([low1_dose[i], full_dose[i], gen_full_dose[i]])
            fake_imgs = self.transfer_display_window(fake_imgs, ref_img=full_dose[i:i+1])
            fake_imgs = torch.from_numpy(fake_imgs).float() / 255.0
            fake_imgs = fake_imgs.transpose(1, 0).reshape((-1, c, w, h))
            self.logger.save_image(
                torchvision.utils.make_grid(fake_imgs, nrow=3),
                n_iter,
                f'{file_id}_{slice_num}'
            )
            self.logger.save_tif(
                gen_full_dose[i:i+1],
                n_iter,
                f'{file_id}_{slice_num}',
                ref_img=full_dose[i:i+1]
            )


# 初始化数据和模型
data_root = '../data_preprocess/gen_data/spect_8cg_npy'
patient_ids = ['001','002','004','005','011','012','018','019','022','025',
               '051','052','054','055','061','062','068','069','072','075',
               '101','102','104','105','111','112','118','119','122','125',
               '151','152','154','155','161','162','168','169','172','175',
               '201','202','204','205','211','212','218','219','222','225',
               '251','252','254','255','261','262','268','269','272','275',
               '301','302','304','305','311','312','318','319','322','325',
               '351','352','354','355','361','362','368','369','372','375']
dataset = SpectDataLoader(data_root, patient_ids)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

test_images = [dataset[i] for i in range(len(dataset))]
low_dose = torch.stack([torch.from_numpy(x[0]) for x in test_images], dim=0).cuda()
full_dose = torch.stack([torch.from_numpy(x[1]) for x in test_images], dim=0).cuda()
forgen_test_images = (low_dose, full_dose)

denoise_fn = Network(in_channels=3, context=True)
ema_model = Diffusion(
    denoise_fn=denoise_fn,
    image_size=36,
    timesteps=1000,
    context=True
).cuda()
logger = LoggerX(save_root=osp.join(
    osp.dirname(osp.dirname(osp.abspath(__file__))), 'output', 'corediff_8cgspect_1000t'
))

# 运行测试
model = Model(test_loader, ema_model, logger, forgen_test_images)
start_time=time.perf_counter()
model.test()
end_time = time.perf_counter()
elapsed_time = end_time - start_time
model.generate_images()

# 处理 TIF 文件
input_folder = "../output/corediff_8cgspect_1000t/save_TIF"
output_folder = "../output/corediff_8cgspect_1000t/save_tif3d"
process_tif_files(input_folder, output_folder)
print(f"预测推理总时间为：{elapsed_time:.6f} 秒")