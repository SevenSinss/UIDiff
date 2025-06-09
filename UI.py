import streamlit as st
import os
import shutil
import tifffile
import os.path as osp
import numpy as np
from natsort import natsorted
from glob import glob
from PIL import Image
import torch
import torchvision
import tqdm
import time
import re
from utils.measure import compute_measure
from utils.loggerx import LoggerX
from models.corediff.corediff_wrapper import Network
from models.corediff.diffusion_modules import Diffusion
from utils.ima_tif import process_tif_files

# 定义自定义 collate_fn
def custom_collate(batch):
    """
    自定义 collate_fn，允许 batch 中包含 None 值
    """
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    inputs = torch.utils.data.dataloader.default_collate(inputs)
    return inputs, targets
# 设置页面布局为宽屏模式
st.set_page_config(layout="wide")

# 定义临时文件夹路径
TEMP_DIR = "temp_uploads"
GATED_DIR = osp.join(TEMP_DIR, "gated")
STATIC_DIR = osp.join(TEMP_DIR, "static")
GATED_SLICES_DIR = osp.join(TEMP_DIR, "gated_slices")
STATIC_SLICES_DIR = osp.join(TEMP_DIR, "static_slices")
NPY_DIR = osp.join(TEMP_DIR, "npy")
OUTPUT_TIF_DIR = osp.join(TEMP_DIR, "output_tif")
OUTPUT_TIF3D_DIR = osp.join(TEMP_DIR, "output_tif3d")

# 创建临时文件夹
for directory in [TEMP_DIR, GATED_DIR, STATIC_DIR, GATED_SLICES_DIR, STATIC_SLICES_DIR, NPY_DIR, OUTPUT_TIF_DIR, OUTPUT_TIF3D_DIR]:
    if not osp.exists(directory):
        os.makedirs(directory)

# 定义SpectDataLoader类
class SpectDataLoader:
    def __init__(self, data_root, dose='8cg'):
        self.data_root = data_root
        self.dose = dose
        self.base_input, self.base_target = self.process_data()

    def process_data(self):
        base_input, base_target = [], []
        input_paths = natsorted(glob(osp.join(self.data_root, 'gated_slice_*.npy')))
        target_paths = natsorted(glob(osp.join(self.data_root, 'static_slice_*.npy')))
        for i in range(len(input_paths)):
            if i > 0 and i < len(input_paths) - 1:
                cat_input = f"{input_paths[i-1]}~{input_paths[i]}~{input_paths[i+1]}"
                base_input.append(cat_input)
                # 允许 target_paths 为空
                if i < len(target_paths):
                    base_target.append(target_paths[i])
                else:
                    base_target.append(None)
        return base_input, base_target

    def __getitem__(self, index):
        input_paths, target_path = self.base_input[index], self.base_target[index]
        input_imgs = [np.load(img_path)[np.newaxis, ...].astype(np.float32) for img_path in input_paths.split('~')]
        input = np.concatenate(input_imgs, axis=0)
        target = np.load(target_path)[np.newaxis, ...].astype(np.float32) if target_path else None
        if target is not None:
            NMSE_scale = np.sum(target) / np.sum(input) if np.sum(input) != 0 else 1.0
            input = input * NMSE_scale
        input = self.normalize_(input)
        return input, target

    def __len__(self):
        # 如果 base_input 不为空，返回其长度
        return len(self.base_input) if any(self.base_input) else 0

    def normalize_(self, img):
        MIN_B = np.min(img)
        MAX_B = np.max(img)
        if MAX_B <= MIN_B:
            return img
        img[img < MIN_B] = MIN_B
        img[img > MAX_B] = MAX_B
        return (img - MIN_B) / (MAX_B - MIN_B)

# 定义Model类
class Model:
    def __init__(self, test_loader, ema_model, logger, test_images, output_tif_dir, output_tif3d_dir, dose, timesteps, progress_bar=None):
        self.test_loader = test_loader
        self.ema_model = ema_model
        self.logger = logger
        self.test_images = test_images
        self.output_tif_dir = output_tif_dir
        self.output_tif3d_dir = output_tif3d_dir
        self.dose = dose
        self.T = timesteps
        self.sampling_routine = "ddim"
        self.start_adjust_iter = 1
        self.context = True
        self.progress_bar = progress_bar

    @torch.no_grad()
    def test(self):
        best_model_path = osp.join('Diffmodel', self.dose, f'{self.T}t', 'best_model')
        try:
            self.ema_model.load_state_dict(torch.load(best_model_path))
            st.write(f"已加载模型 {best_model_path} 进行降噪。")
        except FileNotFoundError:
            st.error(f"模型文件 {best_model_path} 未找到。")
            return
        self.ema_model.eval()
        n_iter = 150000
        psnr, ssim, nmse = 0., 0., 0.
        valid_batches = 0
        total_batches = len(self.test_loader)

        for i, (low_dose, full_dose) in enumerate(tqdm.tqdm(self.test_loader, desc='测试中')):
            low_dose = low_dose.cuda()
            # full_dose 可能为 None 或包含 None 的列表
            if isinstance(full_dose, list) and all(f is None for f in full_dose):
                full_dose = None
            else:
                full_dose = torch.stack([f for f in full_dose if f is not None]).cuda() if full_dose else None
            gen_full_dose, _, _ = self.ema_model.sample(
                batch_size=low_dose.shape[0],
                img=low_dose,
                t=self.T,
                sampling_routine=self.sampling_routine,
                start_adjust_iter=self.start_adjust_iter,
            )
            if full_dose is not None:
                full_dose_norm = self.normalize(full_dose)
                gen_full_dose_norm = self.normalize(gen_full_dose)
                data_range = np.max(full_dose_norm) - np.min(full_dose_norm)
                psnr_score, ssim_score, nmse_score = compute_measure(full_dose_norm, gen_full_dose_norm, data_range=data_range)
                psnr += psnr_score
                ssim += ssim_score
                nmse += nmse_score
                valid_batches += 1
            # 更新进度条（测试阶段：0% 到 50%）
            if self.progress_bar is not None:
                progress = 0.5 * (i + 1) / total_batches
                self.progress_bar.progress(min(progress, 0.5))

        if valid_batches > 0:
            psnr /= valid_batches
            ssim /= valid_batches
            nmse /= valid_batches
            st.write(f"降噪指标: PSNR={psnr:.4f}, SSIM={ssim:.4f}, NMSE={nmse:.4f}")
            self.logger.msg([psnr, ssim, nmse], n_iter)
            self.logger.record_metrics(n_iter, psnr, ssim, nmse)
        else:
            st.warning("未上传静态 SPECT 图像，无法计算 PSNR、SSIM、NMSE 指标。")

    def normalize(self, img):
        img_np = img.cpu().numpy() if isinstance(img, torch.Tensor) else img
        MIN_B = np.min(img_np)
        MAX_B = np.max(img_np)
        if MAX_B <= MIN_B:
            return img_np
        img_np = (img_np - MIN_B) / (MAX_B - MIN_B)
        return img_np

    def transfer_display_window(self, img, ref_img=None, cut_min=None, cut_max=None):
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if isinstance(ref_img, torch.Tensor):
            ref_img = ref_img.cpu().numpy()

        if ref_img is not None:
            MIN_B = np.min(ref_img)
            MAX_B = np.max(ref_img)
            if MAX_B <= MIN_B:
                st.warning(f"参考图像范围无效 ({MIN_B:.6f}, {MAX_B:.6f})，使用默认范围 [0, 1]")
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
            img = (img - (cut_min if cut_min is not None else MIN_B)) / ((cut_max if cut_max is not None else MAX_B) - MIN_B)

        img = 255 * img
        return img

    @torch.no_grad()
    def generate_images(self):
        best_model_path = osp.join('Diffmodel', self.dose, f'{self.T}t', 'best_model')
        try:
            self.ema_model.load_state_dict(torch.load(best_model_path))
            st.write(f"已加载模型 {best_model_path} 进行图像生成。")
        except FileNotFoundError:
            st.error(f"模型文件 {best_model_path} 未找到。")
            return
        self.ema_model.eval()
        n_iter = 150000

        low_dose, full_dose = self.test_images
        total_slices = low_dose.shape[0]

        gen_full_dose, direct_recons, imstep_imgs = self.ema_model.sample(
            batch_size=low_dose.shape[0],
            img=low_dose,
            t=self.T,
            sampling_routine=self.sampling_routine,
            n_iter=n_iter,
            start_adjust_iter=self.start_adjust_iter,
        )

        st.write(f"已生成图像：处理了 {gen_full_dose.shape[0]} 张切片。")

        for i in range(gen_full_dose.shape[0]):
            input_file_path = self.test_loader.dataset.base_input[i]
            context_frames = input_file_path.split('~')
            first_frame_path = context_frames[1]
            base_filename = osp.basename(first_frame_path)
            match = re.match(r'^(gated_slice)_(\d+)\.npy$', base_filename)
            if not match:
                st.warning(f"文件名格式无效：{base_filename}")
                continue
            file_prefix, slice_num = match.groups()

            # 如果没有 full_dose，仅保存输入和生成图像
            if full_dose is not None:
                fake_imgs = torch.stack([low_dose[i, 1].unsqueeze(0), full_dose[i], gen_full_dose[i]])
                ref_img = full_dose[i:i+1]
            else:
                fake_imgs = torch.stack([low_dose[i, 1].unsqueeze(0), gen_full_dose[i]])
                ref_img = None
            fake_imgs = self.transfer_display_window(fake_imgs, ref_img=ref_img)
            fake_imgs = torch.from_numpy(fake_imgs).float() / 255.0
            fake_imgs = fake_imgs.transpose(1, 0).reshape((-1, 1, fake_imgs.shape[2], fake_imgs.shape[3]))
            self.logger.save_image(
                torchvision.utils.make_grid(fake_imgs, nrow=fake_imgs.shape[1]),
                n_iter,
                f'denoised_{slice_num}'
            )
            self.logger.save_tif(
                gen_full_dose[i:i+1],
                n_iter,
                f'denoised_{slice_num}',
                ref_img=ref_img
            )
            # 更新进度条（图像生成阶段：50% 到 95%）
            if self.progress_bar is not None:
                progress = 0.5 + 0.45 * (i + 1) / total_slices
                self.progress_bar.progress(min(progress, 0.95))

        # 处理TIF文件生成3D TIF
        process_tif_files(self.output_tif_dir, self.output_tif3d_dir)
        # 更新进度条到 100%
        if self.progress_bar is not None:
            self.progress_bar.progress(1.0)
        st.success(f"降噪后的3D TIF文件已保存到 {self.output_tif3d_dir}")

# 创建两列布局：左侧选择框和上传，右侧主界面
col1, col2 = st.columns([1, 4])

# 左侧选择框和文件上传
with col1:
    st.header("参数选择与数据上传")

    gating_type = st.selectbox("门控类型选择", options=[8, 16])
    diffusion_steps = st.selectbox("扩散步长选择", options=[5, 10, 100, 1000])

    st.subheader("上传门控SPECT图像")
    gated_spect_file = st.file_uploader("选择门控SPECT图像文件", type=["tif"])
    if gated_spect_file is not None:
        file_path = osp.join(GATED_DIR, gated_spect_file.name)
        with open(file_path, "wb") as f:
            f.write(gated_spect_file.getbuffer())
        st.success(f"门控SPECT图像 {gated_spect_file.name} 上传成功！")

    st.subheader("上传静态SPECT图像（可选）")
    static_spect_file = st.file_uploader("选择静态SPECT图像文件（用于指标计算）", type=["tif"])
    if static_spect_file is not None:
        file_path = osp.join(STATIC_DIR, static_spect_file.name)
        with open(file_path, "wb") as f:
            f.write(gated_spect_file.getbuffer())
        st.success(f"静态SPECT图像 {static_spect_file.name} 上传成功！")

# 右侧主界面
with col2:
    st.title("基于二阶广义扩散模型的心脏门控SPECT影像降噪系统")

    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        if st.button("数据处理"):
            gated_files = [f for f in os.listdir(GATED_DIR) if f.endswith('.tif') and osp.isfile(osp.join(GATED_DIR, f))]
            if not gated_files:
                st.warning("未找到上传的门控SPECT TIF文件！")
            else:
                for file_name in gated_files:
                    file_path = osp.join(GATED_DIR, file_name)
                    try:
                        img = tifffile.imread(file_path)
                        if len(img.shape) >= 3:
                            for z in range(img.shape[0]):
                                slice_path = osp.join(GATED_SLICES_DIR, f"gated_slice_{z}.tif")
                                tifffile.imwrite(slice_path, img[z])
                            st.success(f"{file_name} 已沿Z轴切片并保存到 gated_slices 文件夹！")
                        else:
                            st.warning(f"{file_name} 不是3D图像，跳过切片处理！")
                    except Exception as e:
                        st.error(f"处理 {file_name} 时出错：{str(e)}")

                gated_slice_files = natsorted(glob(osp.join(GATED_SLICES_DIR, '*.tif')))
                if gated_slice_files:
                    for slice_idx, data_path in enumerate(gated_slice_files):
                        try:
                            im = Image.open(data_path)
                            f = np.array(im)
                            f_name = f"gated_slice_{slice_idx}.npy"
                            np.save(osp.join(NPY_DIR, f_name), f.astype(np.uint16))
                        except Exception as e:
                            st.error(f"转换门控切片 {osp.basename(data_path)} 为NPY时出错：{str(e)}")
                    st.success(f"门控切片已转换并保存到 npy 文件夹！")
                else:
                    st.warning("未找到门控SPECT切片文件，无法转换为NPY！")

            static_files = [f for f in os.listdir(STATIC_DIR) if f.endswith('.tif') and osp.isfile(osp.join(STATIC_DIR, f))]
            if static_files:
                for file_name in static_files:
                    file_path = osp.join(STATIC_DIR, file_name)
                    try:
                        img = tifffile.imread(file_path)
                        if len(img.shape) >= 3:
                            for z in range(img.shape[0]):
                                slice_path = osp.join(STATIC_SLICES_DIR, f"static_slice_{z}.tif")
                                tifffile.imwrite(slice_path, img[z])
                            st.success(f"{file_name} 已沿Z轴切片并保存到 static_slices 文件夹！")
                        else:
                            st.warning(f"{file_name} 不是3D图像，跳过切片处理！")
                    except Exception as e:
                        st.error(f"处理 {file_name} 时出错：{str(e)}")

                static_slice_files = natsorted(glob(osp.join(STATIC_SLICES_DIR, '*.tif')))
                if static_slice_files:
                    for slice_idx, data_path in enumerate(static_slice_files):
                        try:
                            im = Image.open(data_path)
                            f = np.array(im)
                            f_name = f"static_slice_{slice_idx}.npy"
                            np.save(osp.join(NPY_DIR, f_name), f.astype(np.uint16))
                        except Exception as e:
                            st.error(f"转换静态切片 {osp.basename(data_path)} 为NPY时出错：{str(e)}")
                    st.success(f"静态切片已转换并保存到 npy 文件夹！")
                else:
                    st.warning("未找到静态SPECT切片文件，无法转换为NPY！")
            else:
                st.info("未上传静态SPECT图像，将跳过指标计算。")

    with col_btn2:
        if st.button("图像降噪"):
            try:
                # 将门控类型映射为 dose
                dose = f"{gating_type}cg"
                dataset = SpectDataLoader(NPY_DIR, dose=dose)
                if len(dataset) == 0:
                    st.error("未找到有效的门控数据，请先运行数据处理！")
                    st.stop()
                # 使用自定义 collate_fn
                test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=custom_collate)
                test_images = [dataset[i] for i in range(len(dataset))]
                low_dose = torch.stack([torch.from_numpy(x[0]) for x in test_images], dim=0).cuda()
                # 检查是否有有效的 full_dose
                if any(x[1] is not None for x in test_images):
                    full_dose = torch.stack([torch.from_numpy(x[1]) for x in test_images if x[1] is not None], dim=0).cuda()
                else:
                    full_dose = None
                test_images = (low_dose, full_dose)

                denoise_fn = Network(in_channels=3, context=True)
                ema_model = Diffusion(
                    denoise_fn=denoise_fn,
                    image_size=36,
                    timesteps=diffusion_steps,
                    context=True
                ).cuda()
                logger = LoggerX(save_root=OUTPUT_TIF_DIR)

                # 初始化进度条
                progress_bar = st.progress(0.0, text="图像降噪处理中...")

                model = Model(test_loader, ema_model, logger, test_images, OUTPUT_TIF_DIR, OUTPUT_TIF3D_DIR, dose, diffusion_steps, progress_bar=progress_bar)

                start_time = time.perf_counter()
                model.test()
                model.generate_images()
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                st.success(f"预测推理总时间为：{elapsed_time:.6f} 秒")
            except Exception as e:
                st.error(f"图像降噪过程中出错：{str(e)}")
                if 'progress_bar' in locals():
                    progress_bar.empty()

    with col_btn3:
        if st.button("清空数据"):
            if osp.exists(TEMP_DIR):
                shutil.rmtree(TEMP_DIR)
                for directory in [TEMP_DIR, GATED_DIR, STATIC_DIR, GATED_SLICES_DIR, STATIC_SLICES_DIR, NPY_DIR, OUTPUT_TIF_DIR, OUTPUT_TIF3D_DIR]:
                    os.makedirs(directory)
                st.success("临时文件夹中的所有数据已清空！")
            else:
                st.warning("临时文件夹已为空！")
