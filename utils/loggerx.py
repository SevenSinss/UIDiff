import torch
import os.path as osp
import os
import time
from torchvision.utils import save_image
import torch.distributed as dist
import inspect
from utils.ops import reduce_tensor, load_network
import pandas as pd
import pydicom
from PIL import Image
import numpy as np


def get_varname(var):
    """
    获取变量名称，从最外层框架向内查找。
    :param var: 要获取名称的变量。
    :return: 字符串
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]


class LoggerX(object):
    def __init__(self, save_root):
        # 定义模型保存和图像保存的目录路径
        self.models_save_dir = osp.join(save_root, 'save_models')
        self.images_save_dir = osp.join(save_root, 'save_images')
        self.IMA_save_dir = osp.join(save_root, 'save_IMA')
        self.TIF_save_dir = osp.join(save_root, 'save_TIF')
        self.metrics_save_dir = osp.join(save_root, 'save_metrics')
        self.metrics_df = pd.DataFrame(columns=['epoch', 'PSNR', 'SSIM', 'NMSE'])

        # 创建保存目录，如果目录已存在则不创建
        os.makedirs(self.models_save_dir, exist_ok=True)
        os.makedirs(self.images_save_dir, exist_ok=True)
        os.makedirs(self.IMA_save_dir, exist_ok=True)
        os.makedirs(self.TIF_save_dir, exist_ok=True)
        os.makedirs(self.metrics_save_dir, exist_ok=True)
        self._modules = []
        self._module_names = []
        self.world_size = 1
        self.local_rank = 0
        # 存储训练过程的损失
        self.train_losses = []
        self.val_losses = []
        self.val_best_loss = float('inf')

    @property
    def modules(self):
        return self._modules

    @property
    def module_names(self):
        return self._module_names

    @modules.setter
    def modules(self, modules):
        for i in range(len(modules)):
            self._modules.append(modules[i])
            self._module_names.append(get_varname(modules[i]))

    def checkpoints(self, epoch):
        if self.local_rank != 0:
            return
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]
            torch.save(module.state_dict(), osp.join(self.models_save_dir, '{}-{}'.format(module_name, epoch)))

    def load_checkpoints(self, epoch):
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]
            module.load_state_dict(load_network(osp.join(self.models_save_dir, '{}-{}'.format(module_name, epoch))))

    def load_test_checkpoints(self, epoch):
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            if module_name == 'ema_model':
                module = self.modules[i]
                module.load_state_dict(load_network(osp.join(self.models_save_dir, '{}-{}'.format(module_name, epoch))))

    def record_metrics(self, epoch, psnr, ssim, nmse):
        new_row = pd.DataFrame([[epoch, psnr, ssim, nmse]], columns=['epoch', 'PSNR', 'SSIM', 'NMSE'])
        self.metrics_df = pd.concat([self.metrics_df, new_row], ignore_index=True)
        self.metrics_df.to_csv(osp.join(self.metrics_save_dir, 'metrics.csv'), index=False)

    def msg(self, stats, step):
        output_str = '[{}] {:05d}, '.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), step)

        for i in range(len(stats)):
            if isinstance(stats, (list, tuple)):
                var = stats[i]
                var_name = get_varname(stats[i])
            elif isinstance(stats, dict):
                var_name, var = list(stats.items())[i]
            else:
                raise NotImplementedError
            if isinstance(var, torch.Tensor):
                var = var.detach().mean()
                var = reduce_tensor(var)
                var = var.item()
            output_str += '{} {:2.5f}, '.format(var_name, var)

        if self.local_rank == 0:
            print(output_str)

    def train_msg(self, stats, step):
        output_str = '[{}] {:05d}, '.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), step)

        for i in range(len(stats)):
            if isinstance(stats, (list, tuple)):
                var = stats[i]
                var_name = get_varname(stats[i])
            elif isinstance(stats, dict):
                var_name, var = list(stats.items())[i]
            else:
                raise NotImplementedError
            if isinstance(var, torch.Tensor):
                var = var.detach().mean()
                var = reduce_tensor(var)
                var = var.item()
            output_str += '{} {:2.5f}, '.format(var_name, var)

        if self.local_rank == 0:
            print(output_str)
            # 提取损失值并保存
            if isinstance(stats, dict):
                tra_loss = stats['loss']
            elif isinstance(stats, (list, tuple)):
                tra_loss = stats[0]
            else:
                raise ValueError("不支持的 stats 类型")

            self.train_losses.append(tra_loss)

    def val_msg(self, stats, step):
        output_str = '[{}] {:05d}, '.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), step)

        for i in range(len(stats)):
            if isinstance(stats, (list, tuple)):
                var = stats[i]
                var_name = get_varname(stats[i])
            elif isinstance(stats, dict):
                var_name, var = list(stats.items())[i]
            else:
                raise NotImplementedError
            if isinstance(var, torch.Tensor):
                var = var.detach().mean()
                var = reduce_tensor(var)
                var = var.item()
            output_str += '{} {:2.5f}, '.format(var_name, var)

        if self.local_rank == 0:
            print(output_str)
            # 提取损失值并保存
            if isinstance(stats, dict):
                val_loss = stats['loss']
            elif isinstance(stats, (list, tuple)):
                val_loss = stats[0]
            else:
                raise ValueError("不支持的 stats 类型")

            self.val_losses.append(val_loss)

    def save_image(self, grid_img, n_iter, sample_type):
        """
        保存图像网格到指定路径。
        :param grid_img: 图像网格，通常是一个张量。
        :param n_iter: 当前训练步骤，用于文件命名。
        :param sample_type: 样本类型，用于文件命名。
        """
        save_path = osp.join(self.images_save_dir, f'{n_iter}_{self.local_rank}_{sample_type}.png')
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        save_image(grid_img, save_path, nrow=1)
        print(f"PNG 图像已保存至 {save_path}")

    def save_IMA(self, img, n_iter, sample_type):
        """
        保存图像为 IMA（DICOM）格式。
        :param img: 要保存的图像，通常是一个张量。
        :param n_iter: 当前训练步骤，用于文件命名。
        :param sample_type: 样本类型，用于文件命名。
        """
        img = img.cpu().numpy()
        img = (img * 65535).astype('uint16')

        dicom_file = pydicom.Dataset()
        dicom_file.PixelData = img.tobytes()
        dicom_file.Rows, dicom_file.Columns = img.shape[-2], img.shape[-1]
        dicom_file.SamplesPerPixel = 1
        dicom_file.PhotometricInterpretation = "MONOCHROME2"

        dicom_file.BitsAllocated = 16
        dicom_file.BitsStored = 16
        dicom_file.HighBit = 15
        dicom_file.PixelRepresentation = 0

        dicom_file.is_little_endian = True
        dicom_file.is_implicit_VR = False

        save_path = osp.join(self.IMA_save_dir, f'{n_iter}_{self.local_rank}_{sample_type}.IMA')
        dicom_file.save_as(save_path)
        print(f"IMA 图像已保存至 {save_path}")

    def save_tif(self, img, n_iter, sample_type, ref_img=None, min_val=None, max_val=None):
        """
        保存图像为 TIF 格式，动态调整图像数据范围以匹配参考图像的物理值范围。
        :param img: 要保存的图像，通常是一个张量，形状为 (1, 1, height, width)，归一化到 [0, 1]。
        :param n_iter: 当前训练步骤，用于文件命名。
        :param sample_type: 样本类型，用于文件命名。
        :param ref_img: 参考图像（例如 full_dose），用于确定物理值范围（优先）。
        :param min_val: 物理值范围的最小值（可选，优先级低于 ref_img）。
        :param max_val: 物理值范围的最大值（可选，优先级低于 ref_img）。
        """
        img = img.cpu().numpy() if isinstance(img, torch.Tensor) else img
        print(f"save_tif: img range before processing: {img.min():.6f} to {img.max():.6f}")

        # 去除批次和通道维度，确保形状为 (height, width)
        if img.ndim == 4:
            img = img[0, 0]
        elif img.ndim == 3:
            img = img[0]
        elif img.ndim != 2:
            raise ValueError(f"Unexpected image shape: {img.shape}")

        # 确定物理值范围
        if ref_img is not None:
            ref_img = ref_img.cpu().numpy() if isinstance(ref_img, torch.Tensor) else ref_img
            if ref_img.ndim == 4:
                ref_img = ref_img[0, 0]
            elif ref_img.ndim == 3:
                ref_img = ref_img[0]
            MIN_B = np.min(ref_img)
            MAX_B = np.max(ref_img)
            if MAX_B <= MIN_B or MAX_B - MIN_B < 1e-6:
                print(f"Warning: Invalid ref_img range ({MIN_B:.6f}, {MAX_B:.6f}), using fallback [0, 65535]")
                MIN_B = 0
                MAX_B = 65535
        elif min_val is not None and max_val is not None:
            MIN_B = min_val
            MAX_B = max_val
            if MAX_B <= MIN_B or MAX_B - MIN_B < 1e-6:
                print(f"Warning: Invalid min_val/max_val range ({MIN_B:.6f}, {MAX_B:.6f}), using fallback [0, 65535]")
                MIN_B = 0
                MAX_B = 65535
        else:
            print("No ref_img or min_val/max_val provided, using fallback [0, 65535]")
            MIN_B = 0
            MAX_B = 65535

        print(f"save_tif: Scaling to range ({MIN_B:.6f}, {MAX_B:.6f})")

        # 恢复到物理值范围
        img = img * (MAX_B - MIN_B) + MIN_B

        # 裁剪到 uint16 范围
        img = np.clip(img, 0, 65535)
        print(f"save_tif: img range after scaling: {img.min():.6f} to {img.max():.6f}")

        # 转换为 uint16
        img = img.astype(np.uint16)
        print(f"save_tif: img range after uint16 conversion: {img.min()} to {img.max()}")

        # 保存为 TIF 文件
        save_path = osp.join(self.TIF_save_dir, f'{n_iter}_{self.local_rank}_{sample_type}.tif')
        img_pil = Image.fromarray(img, mode='I;16')
        img_pil.save(save_path, format='TIFF')
        print(f"TIF 图像已保存至 {save_path}")

    def save_best_model(self, epoch):
        if self.local_rank != 0:
            return
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]
            torch.save(module.state_dict(), osp.join(self.models_save_dir, f'{module_name}-best'))
        print(f"最佳模型已保存于 epoch {epoch}")

    def plot_metrics(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(16, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        save_path = osp.join(self.metrics_save_dir, 'loss_curve.png')
        plt.savefig(save_path)
        plt.close()
        print(f"损失曲线已保存至 {save_path}")