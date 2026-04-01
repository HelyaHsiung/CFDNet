import os.path

import cv2
import torch
from omegaconf import OmegaConf
from data.frames_data_module import FramesDataModule
from data.videos_data_module import VideosDataModule
from model.image_segmentation.UNet import UNet
from model.video_segmentation.CFD_Net import CFD_Net
from model.video_segmentation.Conv_LSTM import ConvLSTM
from model.video_segmentation.MSNet import MSNet
from model.video_segmentation.ST_UNet import ST_UNet
from model.video_segmentation.TDN_Net import TDN_Net
from model.video_segmentation.TEA_Net import TEA_Net
from model.video_segmentation.VSF_Net import VSF_Net
from utils.loss_functions import DiceLoss
from utils.assess import calculate_iou
from utils.render_color import render_input, render_output, render_stacked


def infer():
    device = torch.device('cuda')
    infer_data_name = "simulated_01"
    settings = OmegaConf.load(f"./configs/cfdnet_config.yaml")

    datamodule = VideosDataModule(settings)
    dataloader = datamodule.infer_dataloader(infer_data_name)
    settings.model.pretrained_parameters_path = "E:/Github/CFDNet/log/CFDNet-Gas-Segmentation/2025-01-03_11-07-37/CFDNet_state_dict.pth"
    settings.model.pretrained = True
    model = CFD_Net(settings).to(device)
    model.eval()

    # loss_bce = torch.nn.BCELoss()
    # loss_dice = DiceLoss()

    counter = 0
    cv2.namedWindow("prediction", cv2.WINDOW_AUTOSIZE)
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            x = batch["msi_input"].to(device)                                                   # [B, C, T, H, W] or [B, C, H, W]
            en_x = model.data_aug(x)
            y = model(x)                                                                        # [B*T, C, H, W] or [B, C, H, W]
            pred_mask = (y.squeeze(1).squeeze(0) > 0.5).detach().cpu().numpy()                  # [T, H, W] or       [H, W]
            gt_mask = (batch["gt_output"].squeeze(0).squeeze(0) > 0).detach().cpu().numpy()     # [T, H, W] or       [H, W]
            if isinstance(datamodule, VideosDataModule):
                os.makedirs(f"./output/{infer_data_name}/cfdnet", exist_ok=True)
                for t in range(gt_mask.shape[0]):
                    gas_region = render_output(pred_mask[t, ...])
                    back = render_input(en_x[:, :, t, :, :].squeeze(0).detach().cpu().numpy())
                    output = render_stacked(back, gas_region)
                    cv2.putText(output, f"frame: {counter:03d}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, 0, False)
                    cv2.imwrite(f"./output/{infer_data_name}/cfdnet/{infer_data_name}_pred_{counter:05d}.png", output)
                    cv2.imshow("prediction", output)

                    # print(f"iou of prediction and target is: {calculate_iou(pred_mask[t, ...], gt_mask[t, ...]):8.6f}")
                    # print(f"loss of prediction and target is: {loss_bce(y, batch['gt_output'].transpose(1, 2).squeeze(0).to(y.device)) + loss_dice(y, batch['gt_output'].transpose(1, 2).squeeze(0).to(y.device)):8.6f}")

                    counter = counter + 1
                    key = cv2.waitKey(1000) & 0xFF
                    if key == ord('q'):
                        break
            else:  # isinstance(datamodule, FramesDataModule)
                os.makedirs(f"./output/{infer_data_name}/unet", exist_ok=True)
                gas_region = render_output(pred_mask)
                back = render_input(en_x.squeeze(0).detach().cpu().numpy())
                output = render_stacked(back, gas_region)
                cv2.putText(output, f"frame: {counter:03d}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, 0, False)
                cv2.imwrite(f"./output/{infer_data_name}/unet/{infer_data_name}_pred_{counter:05d}.png", output)
                cv2.imshow("prediction", output)

                # print(f"iou of prediction and target is: {calculate_iou(pred_mask, gt_mask):8.6f}")
                # print(f"loss of prediction and target is: {loss_bce(y, batch['gt_output'].to(y.device)) + loss_dice(y, batch['gt_output'].to(y.device)):8.6f}")

                counter = counter + 1
                key = cv2.waitKey(1000) & 0xFF
                if key == ord('q'):
                    break
    cv2.destroyWindow("prediction")


if __name__ == "__main__":
    infer()
    print("All Job Were Done!")
