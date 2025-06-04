import torch
import numpy as np
from skimage.metrics import structural_similarity
from sklearn.metrics import mean_squared_error

def compute_measure(y, pred, data_range):
    # 如果输入是 PyTorch 张量，转换为 NumPy 数组
    if hasattr(y, 'cpu') and hasattr(y, 'numpy'):
        y = y.cpu().numpy()
    if hasattr(pred, 'cpu') and hasattr(pred, 'numpy'):
        pred = pred.cpu().numpy()

    # 处理 4D 输入 (batch_size, C, H, W)
    if y.ndim == 4:
        if y.shape[0] != 1:
            raise ValueError(f"Batch size must be 1 for single image metrics, got {y.shape[0]}")
        y = y[0]  # 提取第一张图像，变为 (C, H, W)
        pred = pred[0]
    
    # 处理可能的维度
    if y.ndim == 3:  # 多通道图像 (C, H, W)
        if y.shape[0] in {1, 3}:  # 通道在前，PyTorch 格式
            channel_axis = 0
            h, w = y.shape[1:]
        else:  # 通道在后 (H, W, C)，NumPy 格式
            channel_axis = -1
            h, w = y.shape[:-1]
    elif y.ndim == 2:  # 单通道图像 (H, W)
        channel_axis = None
        h, w = y.shape
    else:
        raise ValueError(f"Unsupported image dimensions: {y.shape}")

    # 计算 PSNR
    mse = mean_squared_error(y.flatten(), pred.flatten())
    psnr = 10 * np.log10(1.0 / mse) if mse != 0 else float('inf')

    # 计算 SSIM
    win_size = min(h, w)
    if win_size < 3:
        raise ValueError("Image dimensions too small for SSIM computation (minimum 3x3)")
    if win_size % 2 == 0:
        win_size -= 1
    ssim = structural_similarity(y, pred, data_range=1.0, win_size=win_size, channel_axis=channel_axis)

    # 计算 NMSE
    nmse = mean_squared_error(y.flatten(), pred.flatten()) / np.mean(y ** 2)

    return psnr, ssim, nmse

# def compute_measure(y, pred, data_range):
#     psnr = compute_PSNR(pred, y, data_range)
#     ssim = compute_SSIM(pred, y, data_range)
#     rmse = compute_RMSE(pred, y)
#     return psnr, ssim, rmse

# def compute_MSE(img1, img2):
#     return ((img1/1.0 - img2/1.0) ** 2).mean()
#
#
# def compute_RMSE(img1, img2):
#     # img1 = img1 * 2000 / 255 - 1000
#     # img2 = img2 * 2000 / 255 - 1000
#     if type(img1) == torch.Tensor:
#         return torch.sqrt(compute_MSE(img1, img2)).item()
#     else:
#         return np.sqrt(compute_MSE(img1, img2))
#
#
# def compute_PSNR(img1, img2, data_range):
#     if type(img1) == torch.Tensor:
#         mse_ = compute_MSE(img1, img2)
#         return 10 * torch.log10((data_range ** 2) / mse_).item()
#     else:
#         mse_ = compute_MSE(img1, img2)
#         return 10 * np.log10((data_range ** 2) / mse_)
#
#
# def compute_SSIM(img1, img2, data_range, window_size=11, channel=1, size_average=True):
#     # referred from https://github.com/Po-Hsun-Su/pytorch-ssim
#     if len(img1.shape) == 2:
#         h, w = img1.shape
#         if type(img1) == torch.Tensor:
#             img1 = img1.view(1, 1, h, w)
#             img2 = img2.view(1, 1, h, w)
#         else:
#             img1 = torch.from_numpy(img1[np.newaxis, np.newaxis, :, :])
#             img2 = torch.from_numpy(img2[np.newaxis, np.newaxis, :, :])
#     window = create_window(window_size, channel)
#     window = window.type_as(img1)
#
#     mu1 = F.conv2d(img1, window, padding=window_size//2)
#     mu2 = F.conv2d(img2, window, padding=window_size//2)
#     mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
#     mu1_mu2 = mu1*mu2
#
#     sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2) - mu1_sq
#     sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2) - mu2_sq
#     sigma12 = F.conv2d(img1*img2, window, padding=window_size//2) - mu1_mu2
#
#     C1, C2 = (0.01*data_range)**2, (0.03*data_range)**2
#     #C1, C2 = 0.01**2, 0.03**2
#
#     ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
#     # if size_average:
#     #     return ssim_map.mean().item()
#     # else:
#     #     return ssim_map.mean(1).mean(1).mean(1).item()
#     if size_average:
#         return ssim_map.mean().item()
#     else:
#         return ssim_map.mean(1).mean(1).mean(1).item()
#
#
# def gaussian(window_size, sigma):
#     gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
#     return gauss / gauss.sum()
#
#
# def create_window(window_size, channel):
#     _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
#     _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#     window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
#     return window