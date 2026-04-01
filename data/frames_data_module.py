import os
import cv2
import torch
import pandas as pd
from osgeo import gdal
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
gdal.UseExceptions()


class FramesDataSet(Dataset):
    def __init__(self, csv_path: str):
        super().__init__()
        self.bands = 6
        self.lines = 480
        self.samples = 640
        df = pd.read_csv(csv_path)
        self.msi_input_dataset = df["msi_inputs"].values
        self.gt_output_dataset = df["gt_outputs"].values

        self.root_dir = os.path.join(os.path.dirname(csv_path), "..")
        self.msi_input_data_size = len(self.msi_input_dataset)
        self.gt_output_data_size = len(self.gt_output_dataset)
        if not (self.msi_input_data_size == self.gt_output_data_size):
            raise Exception("Size of inputs and outputs must be same!")

    def __len__(self):
        return self.msi_input_data_size

    def __getitem__(self, index: int):
        """
        msi_input
        """    # (6, H, W)
        msi = gdal.Open(f"{self.root_dir}/{self.msi_input_dataset[index]}")
        image = msi.ReadAsArray(0, 0, msi.RasterXSize, msi.RasterYSize)  # [6, H, W]
        del msi
        msi_input_data = torch.tensor(image).float()
        """
        gt_output
        """    # (1, H, W)
        mask = cv2.imread(f"{self.root_dir}/{self.gt_output_dataset[index]}", cv2.IMREAD_GRAYSCALE)
        gt_output_data = torch.gt(torch.tensor(mask), 0).float().unsqueeze(0)  # [1, H, W]
        return {"msi_input": msi_input_data, "gt_output": gt_output_data}


class FramesDataModule(LightningDataModule):
    def __init__(self, settings: DictConfig):
        super().__init__()
        current_file_path = os.path.dirname(__file__)
        self.dataset_train = FramesDataSet(f"{current_file_path}/csv_frames/train.csv")
        self.dataset_val = FramesDataSet(f"{current_file_path}/csv_frames/val.csv")
        self.dataset_test = FramesDataSet(f"{current_file_path}/csv_frames/test.csv")
        self.dataset_infer_realworld_01 = FramesDataSet(f"{current_file_path}/csv_frames/realworld_01.csv")
        self.dataset_infer_realworld_02 = FramesDataSet(f"{current_file_path}/csv_frames/realworld_02.csv")
        self.dataset_infer_realworld_03 = FramesDataSet(f"{current_file_path}/csv_frames/realworld_03.csv")
        self.dataset_infer_simulated_01 = FramesDataSet(f"{current_file_path}/csv_frames/simulated_01.csv")
        self.dataset_infer_simulated_02 = FramesDataSet(f"{current_file_path}/csv_frames/simulated_02.csv")
        self.dataset_infer_simulated_03 = FramesDataSet(f"{current_file_path}/csv_frames/simulated_03.csv")
        self.dataset_infer_simulated_04 = FramesDataSet(f"{current_file_path}/csv_frames/simulated_04.csv")
        self.dataset_infer_simulated_05 = FramesDataSet(f"{current_file_path}/csv_frames/simulated_05.csv")
        self.dataset_infer_simulated_06 = FramesDataSet(f"{current_file_path}/csv_frames/simulated_06.csv")
        self.dataset_infer_simulated_07 = FramesDataSet(f"{current_file_path}/csv_frames/simulated_07.csv")
        self.dataset_infer_simulated_08 = FramesDataSet(f"{current_file_path}/csv_frames/simulated_08.csv")
        self.dataset_infer_simulated_09 = FramesDataSet(f"{current_file_path}/csv_frames/simulated_09.csv")
        self.dataset_infer_simulated_10 = FramesDataSet(f"{current_file_path}/csv_frames/simulated_10.csv")
        self.dataset_infer_simulated_11 = FramesDataSet(f"{current_file_path}/csv_frames/simulated_11.csv")
        self.dataset_infer_simulated_12 = FramesDataSet(f"{current_file_path}/csv_frames/simulated_12.csv")

        self.batch_size = settings.dataloader.batch_size
        self.num_workers = settings.dataloader.num_workers

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def infer_dataloader(self, infer_video_name):
        dataset = getattr(self, f"dataset_infer_{infer_video_name}")
        return DataLoader(dataset, batch_size=1, num_workers=self.num_workers, persistent_workers=True)
