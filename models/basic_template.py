# This part builds heavily on https://github.com/Hzzone/DU-GAN.
import torch
import os.path as osp
import tqdm
import argparse
import torch.distributed as dist
import numpy as np
from utils.dataset import dataset_dict
from utils.loggerx import LoggerX
from utils.sampler import RandomSampler
from utils.ops import load_network

#主要参数和运行训练、测试
class TrainTask(object):

    def __init__(self, opt):
        self.opt = opt
        self.logger = LoggerX(save_root=osp.join(
            osp.dirname(osp.dirname(osp.abspath(__file__))), 'output', '{}_{}'.format(opt.model_name, opt.run_name)))
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        self.set_loader()
        self.set_model()

    @staticmethod
    def build_default_options():
        parser = argparse.ArgumentParser('Default arguments for training of different methods')

        parser.add_argument('--save_freq', type=int, default=2500,
                            help='save frequency')
        parser.add_argument('--batch_size', type=int, default=4,
                            help='batch_size')
        parser.add_argument('--test_batch_size', type=int, default=1,
                            help='test_batch_size')
        parser.add_argument('--num_workers', type=int, default=4,
                            help='num of workers to use')
        parser.add_argument('--max_iter', type=int, default=150000,
                            help='number of training iterations')
        parser.add_argument('--resume_iter', type=int, default=0,
                            help='number of training epochs')
        parser.add_argument('--test_iter', type=int, default=150000,
                            help='number of epochs for test')
        parser.add_argument("--local_rank", type=int, default=0)
        parser.add_argument("--mode", type=str, default='train')
        parser.add_argument('--wandb', action="store_true")

        # run_name and model_name
        parser.add_argument('--run_name', type=str, default='default',
                            help='each run name')
        parser.add_argument('--model_name', type=str, default='corediff',
                            help='the type of method')

        # training parameters for one-shot learning framework
        parser.add_argument("--osl_max_iter", type=int, default=3001,
                            help='number of training iterations for one-shot learning framework training')
        parser.add_argument("--osl_batch_size", type=int, default=8,
                            help='batch size for one-shot learning framework training')
        parser.add_argument("--index", type=int, default=10,
                            help='slice index selected for one-shot learning framework training')
        parser.add_argument("--unpair", action="store_true",
                            help='use unpaired data for one-shot learning framework training')
        parser.add_argument("--patch_size", type=int, default=256,
                            help='patch size used to divide the image')

        # dataset
        parser.add_argument('--train_dataset', type=str, default='mayo_2016_sim')
        parser.add_argument('--test_dataset', type=str, default='mayo_2016_sim')   # mayo_2020, piglte, phantom, mayo_2016
        parser.add_argument('--test_ids', type=str, default='9',
                            help='test patient index for Mayo 2016')
        parser.add_argument('--context', action="store_true",
                            help='use contextual information')   #
        parser.add_argument('--image_size', type=int, default=512)
        parser.add_argument('--dose', type=str, default=5,
                            help='dose% data use for training and testing')

        return parser

    @staticmethod
    def build_options():
        pass

    def load_pretrained_dict(self, file_name: str):
        self.project_root = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
        return load_network(osp.join(self.project_root, 'pretrained', file_name))
    #加载数据，数据定义
    def set_loader(self):
        opt = self.opt

        if opt.mode == 'train':
            test_ids = list(map(int, opt.test_ids.split(',')))  # 解析test_ids为整数列表
            train_dataset = dataset_dict['train'](
                dataset=opt.train_dataset,
                test_ids=test_ids,
                dose=opt.dose,
                context=opt.context,
            )
            train_sampler = RandomSampler(dataset=train_dataset, batch_size=opt.batch_size,
                                          num_iter=opt.max_iter,
                                          restore_iter=opt.resume_iter)

            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=opt.batch_size,
                sampler=train_sampler,
                shuffle=False,
                drop_last=False,
                num_workers=opt.num_workers,
                pin_memory=True
            )
            self.train_loader = train_loader

        test_dataset = dataset_dict[opt.test_dataset](
            dataset=opt.test_dataset,
            test_ids=test_ids,
            dose=opt.dose,
            context=opt.context
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=opt.test_batch_size,
            shuffle=False,
            num_workers=opt.num_workers,
            pin_memory=True
        )
        self.test_loader = test_loader

        # test_images = [test_dataset[i] for i in range(0, min(300, len(test_dataset)), 75)]
        # 加载全部测试图像用于测试和可视化
        test_images = [test_dataset[i] for i in range(len(test_dataset))]
        low_dose = torch.stack([torch.from_numpy(x[0]) for x in test_images], dim=0).cuda()
        full_dose = torch.stack([torch.from_numpy(x[1]) for x in test_images], dim=0).cuda()
        self.test_images = (low_dose, full_dose)

        self.test_dataset = test_dataset

    #训练、测试
    def fit(self):
        opt = self.opt
        if opt.mode == 'train':
            if opt.resume_iter > 0:
                self.logger.load_checkpoints(opt.resume_iter)

            # training routine
            loader = iter(self.train_loader)
            for n_iter in tqdm.trange(opt.resume_iter + 1, opt.max_iter + 1, disable=(self.rank != 0)):
                inputs = next(loader)
                self.train(inputs, n_iter)
                if n_iter % opt.save_freq == 0:
                    self.logger.checkpoints(n_iter)
                    self.test(n_iter)
                    # self.generate_images(n_iter)

        elif opt.mode == 'test':
            self.logger.load_test_checkpoints(opt.test_iter)
            self.test(opt.test_iter)
            # self.generate_images(opt.test_iter)

        # train one-shot learning framework
        elif opt.mode == 'train_osl_framework':
            self.logger.load_test_checkpoints(opt.test_iter)
            self.train_osl_framework(opt.test_iter)

        # test one-shot learning framework
        elif opt.mode == 'test_osl_framework':
            self.logger.load_test_checkpoints(opt.test_iter)
            self.test_osl_framework(opt.test_iter)


    def set_model(opt):
        pass

    def train(self, inputs, n_iter):
        pass

    @torch.no_grad()
    def test(self, n_iter):
        pass

    @torch.no_grad()
    def generate_images(self, n_iter):
        pass

    def adjust_learning_rate(self, n_iter):
        opt = self.opt
        pass

    def transfer_calculate_window(self, img, ref_img=None, for_display=True, cut_min=None, cut_max=None):
        """
        将归一化图像恢复到物理值范围，并根据需要裁剪或映射用于计算或显示。

        Args:
            img: 归一化图像（Tensor 或 ndarray，范围 [0, 1]）。
            ref_img: 参考图像（例如 full_dose），用于确定物理值范围（可选）。
            for_display: 如果 True，映射到 [0, 255] 用于显示；如果 False，保留物理值用于计算。
            cut_min: 裁剪的最小值（可选，默认为 ref_img 的最小值或 0）。
            cut_max: 裁剪的最大值（可选，默认为 ref_img 的最大值）。

        Returns:
            处理后的图像（Tensor 或 ndarray）。
        """
        # 转换为 numpy 数组以便处理
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if isinstance(ref_img, torch.Tensor):
            ref_img = ref_img.cpu().numpy()

        # 动态确定物理值范围
        if ref_img is not None:
            MIN_B = np.min(ref_img)
            MAX_B = np.max(ref_img)
        else:
            MIN_B = 0  # 回退值，可根据实际需求调整
            MAX_B = np.max(img) if np.max(img) > 0 else 1.0  # 避免除以零

        # 恢复到物理值范围
        img = img * (MAX_B - MIN_B) + MIN_B

        # 裁剪
        if cut_min is None:
            cut_min = MIN_B
        if cut_max is None:
            cut_max = MAX_B
        img = np.clip(img, cut_min, cut_max)

        if for_display:
            # 映射到 [0, 255] 用于显示
            if cut_max == cut_min:
                img = np.zeros_like(img)  # 避免除以零
            else:
                img = 255 * (img - cut_min) / (cut_max - cut_min)
            return img.astype(np.uint8)
        else:
            # 保留物理值用于计算
            return img

    def transfer_display_window(self, img, MIN_B=0, MAX_B=65535, cut_min=0, cut_max=65535):
        img = img * (MAX_B - MIN_B) + MIN_B  # 将归一化图像恢复到物理值范围 [MIN_B, MAX_B]
        img[img < cut_min] = cut_min  # 将低于 cut_min 的像素值裁剪为 cut_min
        img[img > cut_max] = cut_max  # 将高于 cut_max 的像素值裁剪为 cut_max
        # 将图像值重新映射到 [0, 1] 的范围，用于显示或进一步处理
        img = (img - cut_min) / (cut_max - cut_min)
        return img  # 返回去归一化并且适合显示的图像


