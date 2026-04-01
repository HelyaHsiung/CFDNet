import os
import cv2
import torch
import pandas as pd
from osgeo import gdal
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
gdal.UseExceptions()


class VideosDataSet(Dataset):
    def __init__(self, csv_path: str, time_steps: int = 8):
        super().__init__()
        self.bands = 6
        self.lines = 480
        self.samples = 640
        self.time_steps = time_steps
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
        """    # (Channel, Time, Height, Width)
        msi_input_data = torch.zeros((self.bands, self.time_steps, self.lines, self.samples)).float()  # [6, 8, H, W]
        msi_inputs = self.msi_input_dataset[index].split(" ")
        for t in range(self.time_steps):
            msi = gdal.Open(f"{self.root_dir}/{msi_inputs[t]}")
            image = msi.ReadAsArray(0, 0, msi.RasterXSize, msi.RasterYSize)
            del msi
            msi_input_data[:, t, :, :] = torch.tensor(image).float()
        """
        gt_output
        """    # (1, Time, Height, Width)
        gt_output_data = torch.zeros((1, self.time_steps, self.lines, self.samples)).float()
        gt_outputs = self.gt_output_dataset[index].split(" ")
        for t in range(self.time_steps):
            mask = cv2.imread(f"{self.root_dir}/{gt_outputs[t]}", cv2.IMREAD_GRAYSCALE)
            gt_output_data[:, t, :, :] = torch.gt(torch.tensor(mask), 0).float().unsqueeze(0)

        return {"msi_input": msi_input_data, "gt_output": gt_output_data}


class VideosDataModule(LightningDataModule):
    def __init__(self, settings: DictConfig):
        super().__init__()
        current_file_path = os.path.dirname(__file__)
        self.dataset_train = VideosDataSet(f"{current_file_path}/csv_videos/train.csv", settings.model.time_steps)
        self.dataset_val = VideosDataSet(f"{current_file_path}/csv_videos/val.csv", settings.model.time_steps)
        self.dataset_test = VideosDataSet(f"{current_file_path}/csv_videos/test.csv", settings.model.time_steps)
        self.dataset_infer_realworld_01 = VideosDataSet(f"{current_file_path}/csv_videos/realworld_01.csv", settings.model.time_steps)
        self.dataset_infer_realworld_02 = VideosDataSet(f"{current_file_path}/csv_videos/realworld_02.csv", settings.model.time_steps)
        self.dataset_infer_realworld_03 = VideosDataSet(f"{current_file_path}/csv_videos/realworld_03.csv", settings.model.time_steps)
        self.dataset_infer_simulated_01 = VideosDataSet(f"{current_file_path}/csv_videos/simulated_01.csv", settings.model.time_steps)
        self.dataset_infer_simulated_02 = VideosDataSet(f"{current_file_path}/csv_videos/simulated_02.csv", settings.model.time_steps)
        self.dataset_infer_simulated_03 = VideosDataSet(f"{current_file_path}/csv_videos/simulated_03.csv", settings.model.time_steps)
        self.dataset_infer_simulated_04 = VideosDataSet(f"{current_file_path}/csv_videos/simulated_04.csv", settings.model.time_steps)
        self.dataset_infer_simulated_05 = VideosDataSet(f"{current_file_path}/csv_videos/simulated_05.csv", settings.model.time_steps)
        self.dataset_infer_simulated_06 = VideosDataSet(f"{current_file_path}/csv_videos/simulated_06.csv", settings.model.time_steps)
        self.dataset_infer_simulated_07 = VideosDataSet(f"{current_file_path}/csv_videos/simulated_07.csv", settings.model.time_steps)
        self.dataset_infer_simulated_08 = VideosDataSet(f"{current_file_path}/csv_videos/simulated_08.csv", settings.model.time_steps)
        self.dataset_infer_simulated_09 = VideosDataSet(f"{current_file_path}/csv_videos/simulated_09.csv", settings.model.time_steps)
        self.dataset_infer_simulated_10 = VideosDataSet(f"{current_file_path}/csv_videos/simulated_10.csv", settings.model.time_steps)
        self.dataset_infer_simulated_11 = VideosDataSet(f"{current_file_path}/csv_videos/simulated_11.csv", settings.model.time_steps)
        self.dataset_infer_simulated_12 = VideosDataSet(f"{current_file_path}/csv_videos/simulated_12.csv", settings.model.time_steps)

        self.batch_size = settings.dataloader.batch_size
        self.num_workers = settings.dataloader.num_workers

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.dataset_infer, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def infer_dataloader(self, infer_video_name):
        dataset = getattr(self, f"dataset_infer_{infer_video_name}")
        return DataLoader(dataset, batch_size=1, num_workers=self.num_workers, persistent_workers=True)
