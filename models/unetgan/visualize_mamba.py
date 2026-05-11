import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model


def tensor_to_gray(img_tensor):
    """
    输入:
        (1,1,H,W) 或 (1,H,W)
    输出:
        numpy float, shape=(H,W), 值域[0,1]
    """
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]
    if img_tensor.dim() == 3:
        img_tensor = img_tensor[0]

    img = img_tensor.detach().float().cpu().numpy()
    img = (img + 1.0) / 2.0
    img = np.clip(img, 0, 1)
    return img


def upsample_score_map(score_map, target_h, target_w, apply_sigmoid=True):
    """
    score_map: (1,1,h,w)
    return:    (target_h, target_w)
    """
    if apply_sigmoid:
        score_map = torch.sigmoid(score_map)

    score_map = torch.nn.functional.interpolate(
        score_map,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False
    )

    score_map = score_map[0, 0].detach().float().cpu().numpy()
    return score_map


# def save_ir_overlay(ir_img, pred_mamba, save_path, alpha=0.78, base_darkness=0.18):
#     """
#     ir_img:        (1,1,H,W)   红外底图
#     pred_mamba:    (1,1,h,w)   D_mamba 输出
#     alpha:         热图透明度
#     base_darkness: 底图压暗系数，越小背景越不明显
#     """
#     # 1) 红外底图
#     ir_base = tensor_to_gray(ir_img)   # (H, W), [0,1]
#     H, W = ir_base.shape
#
#     # 2) 底图压暗，弱化背景
#     ir_base = np.clip(ir_base * base_darkness, 0, 1)
#
#     # 3) 判别器输出上采样
#     heatmap = upsample_score_map(pred_mamba, H, W, apply_sigmoid=True)
#
#     # 4) 归一化：压低低响应，拉开高响应
#     q_low = np.percentile(heatmap, 20)
#     q_high = np.percentile(heatmap, 99)
#     heatmap = (heatmap - q_low) / (q_high - q_low + 1e-8)
#     heatmap = np.clip(heatmap, 0, 1)
#
#     # 5) 进一步强化高响应，让高注意力更偏红
#     heatmap = np.power(heatmap, 1.8)
#
#     fig, ax = plt.subplots(1, 1, figsize=(8, 8))
#
#     # 底图：暗红外
#     ax.imshow(ir_base, cmap="gray", vmin=0, vmax=1)
#
#     # 叠加：红色系热图
#     ax.imshow(heatmap, cmap="jet", alpha=alpha, vmin=0, vmax=1)
#
#     ax.axis("off")
#
#     save_dir = os.path.dirname(save_path)
#     if save_dir != "":
#         os.makedirs(save_dir, exist_ok=True)
#
#     plt.savefig(save_path, dpi=240, bbox_inches="tight", pad_inches=0)
#     plt.close(fig)
#     print(f"[OK] saved to: {save_path}")
def save_ir_overlay(ir_img, pred_mamba, save_path, alpha=0.78, base_darkness=0.18):
    """
    ir_img:        (1,1,H,W)   红外底图
    pred_mamba:    (1,1,h,w)   D_mamba 输出
    alpha:         热图透明度
    base_darkness: 底图压暗系数
    """
    # 1) 红外底图
    ir_base = tensor_to_gray(ir_img)
    H, W = ir_base.shape

    # 2) 底图压暗
    ir_base = np.clip(ir_base * base_darkness, 0, 1)

    # 3) 判别器输出上采样
    heatmap = upsample_score_map(pred_mamba, H, W, apply_sigmoid=True)

    # 4) 归一化：稍微提高低值截断，减少整片蓝背景
    q_low = np.percentile(heatmap, 30)
    q_high = np.percentile(heatmap, 99)
    heatmap = (heatmap - q_low) / (q_high - q_low + 1e-8)
    heatmap = np.clip(heatmap, 0, 1)

    # 5) 保留一定高响应强化
    heatmap = np.power(heatmap, 1.6)

    # 6) 截断 jet 的前段深蓝，只保留后面较柔和的蓝-青-绿-黄-红
    jet_soft = LinearSegmentedColormap.from_list(
        "jet_soft",
        plt.cm.jet(np.linspace(0.22, 1.0, 256))
    )

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # 底图
    ax.imshow(ir_base, cmap="gray", vmin=0, vmax=1)

    # 热图
    ax.imshow(heatmap, cmap=jet_soft, alpha=alpha, vmin=0, vmax=1)

    ax.axis("off")

    save_dir = os.path.dirname(save_path)
    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)

    plt.savefig(save_path, dpi=240, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"[OK] saved to: {save_path}")



def main():
    opt = TrainOptions().parse()

    # =========================
    # 只做可视化，不做训练
    # =========================
    opt.continue_train = True
    opt.phase = "test"
    opt.serial_batches = True
    opt.no_flip = True

    # ===== 你自己的工程路径 =====
    opt.dataroot = "./datasets/AVIID-1"
    opt.dataset_mode = "AVIID_1"
    opt.name = "physmamba_AVIID_1"
    opt.checkpoints_dir = "./checkpoints"
    opt.pretrained_encoder_path = "./pretrained/encoders_best.pth"

    # ===== 保存目录 =====
    results_dir = "./results_vis"

    # ===== 最多可视化多少张 =====
    max_vis = 500

    print("========== Visualization Config ==========")
    print("dataroot:", opt.dataroot)
    print("dataset_mode:", opt.dataset_mode)
    print("name:", opt.name)
    print("which_epoch:", opt.which_epoch)
    print("checkpoints_dir:", opt.checkpoints_dir)
    print("phase:", opt.phase)
    print("results_dir:", results_dir)
    print("max_vis:", max_vis)
    print("==========================================")

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    model = create_model(opt)

    if hasattr(model, "eval"):
        model.eval()

    save_root = os.path.join(
        results_dir,
        opt.name,
        f"mamba_ir_overlay_epoch_{opt.which_epoch}"
    )
    os.makedirs(save_root, exist_ok=True)

    for i, data in enumerate(dataset):
        if i >= max_vis:
            break

        model.set_input(data)

        with torch.no_grad():
            model.real_A = model.input_A
            model.real_B = model.input_B
            model.fake_B = model.netG(model.real_A)

            # 与训练时一致
            fake_AB = torch.cat((model.real_A, model.fake_B), dim=1)
            pred_fake_mamba = model.netD_thermal_mamba(fake_AB)

        img_name = f"{i:04d}.png"
        if isinstance(data, dict) and "A_paths" in data:
            p = data["A_paths"]
            if isinstance(p, (list, tuple)) and len(p) > 0:
                img_name = os.path.basename(p[0])
            elif isinstance(p, str):
                img_name = os.path.basename(p)

        base_name, _ = os.path.splitext(img_name)
        save_path = os.path.join(save_root, f"{base_name}_ir_overlay2.png")

        save_ir_overlay(
            ir_img=model.fake_B[:1],  # 底图：生成红外
            pred_mamba=pred_fake_mamba[:1],  # 热图：D_mamba(fake_AB)
            save_path=save_path,
            alpha=0.78,
            base_darkness=0.18
        )


    print("[DONE] visualization finished.")


if __name__ == "__main__":
    main()
