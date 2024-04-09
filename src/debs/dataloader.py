# -------------------------------------------------------
#
#        |
#       / \
#      / _ \                  ESA - PROJECT
#     |.o '.|
#     |'._.'|          BLACK SEA DEOXYGENATION EMULATOR
#     |     |
#   ,'|  |  |`.             BY VICTOR MANGELEER
#  /  |  |  |  \
#  |,-'--|--'-.|                2023-2024
#
#
# -------------------------------------------------------
#
# Documentation
# -------------
# A tool to create a dataloader that processes and loads the Black Sea datasets on the fly
#
import numpy as np

# Torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class BlackSea_Dataloader():
   r"""A tool to create a dataloader that processes and loads the Black Sea datasets on the fly"""

   def __init__(self, x: list,
                      y: np.array,
                      t: np.array,
                   mesh: np.array,
                   mask: np.array,
             bathymetry: np.array,
             window_inp: int   = 1,
          datasets_size: list  = [0.7, 0.2]):

      # Initialization
      timesteps, x_res, y_res = y.shape

      # Stacking all the inputs
      x = np.stack(x, axis = 1)

      # Determining the indexes for each dataset
      train_idx = int(timesteps * datasets_size[0])
      valid_idx = int(timesteps * (datasets_size[0] + datasets_size[1]))

      # Splitting the data
      self.x_train, self.x_validation, self.x_test = x[:train_idx], x[train_idx:valid_idx], x[valid_idx:]
      self.y_train, self.y_validation, self.y_test = y[:train_idx], y[train_idx:valid_idx], y[valid_idx:]

      # Storing other information
      self.time       = t
      self.mesh       = mesh
      self.mask       = np.expand_dims(mask,       axis = 0)
      self.bathymetry = np.expand_dims(bathymetry, axis = 0)
      self.window_inp = window_inp

   def get_number_of_samples(self, type : str):
      r"""Returns the number of samples in a given dataset, i.e. training, validation or test"""
      return getattr(self, f"x_{type}").shape[0] - self.window_inp

   def get_dataloader(self, type: str, num_workers = 0, batch_size: int = 8):
         r"""Creates and returns a dataloader for a given type, i.e. training, validation or test"""

         # Security
         assert type in ["train", "validation", "test"], f"ERROR (BlackSea_Dataloader) Type must be either 'train', 'validation' or 'test' ({type})"

         class BS_Dataset(Dataset):
               r"""Pytorch dataloader"""

               def __init__(self, x: np.array, t: np.array, y: np.array, mesh: np.array, mask: np.array, bathymetry: np.array, window_inp: int):
                  self.x          = x
                  self.t          = t
                  self.y          = y
                  self.mesh       = mesh
                  self.mask       = mask
                  self.bathymetry = bathymetry
                  self.window_inp = window_inp
                  self.x_res      = self.x.shape[2]
                  self.y_res      = self.x.shape[3]
                  self.time_img   = np.ones((self.x_res, self.y_res))

               def process(self, index : int):
                  """Used as a processing pipeline, i.e. it fetch and process a single data sample"""

                  # Determination of the indexes to perform the extraction
                  index_begin  = index
                  index_end    = index + self.window_inp

                  # Extracting the input and output
                  x = self.x[index_begin : index_end]
                  y = self.y[index_end - 1]

                  # Adding missing dimensions
                  y = np.expand_dims(y, axis = (0,1))

                  # Extracting the day at which the prediction is made and creating a map with it
                  t = np.expand_dims(self.time_img * self.t[index_end - 1], axis = 0)

                  # Merging window intputs with physical variables values
                  x = x.reshape((-1, self.x_res, self.y_res))

                  # Masking the useless information (maybe needs to be changed)
                  x = np.where(self.mask == 0, 0, x)

                  # Stacking all the information
                  x = np.concatenate([x, t, self.mesh, self.bathymetry], axis = 0)

                  # Returning the preprocessed samples (format changed for pytorch)
                  return x.astype(np.float32), y.astype(np.float32)

               def __len__(self):
                  return self.x.shape[0] - self.window_inp

               def __getitem__(self, idx):
                  return self.process(index = idx)

         # Creation of the dataset for dataloader
         dataset = BS_Dataset(x          = getattr(self, f"x_{type}"),
                              y          = getattr(self, f"y_{type}"),
                              t          = getattr(self, f"time"),
                              mesh       = getattr(self, f"mesh"),
                              mask       = getattr(self, f"mask"),
                              bathymetry = getattr(self, f"bathymetry"),
                              window_inp = getattr(self, f"window_inp"))

         # Creation of the dataloader
         return DataLoader(dataset, batch_size = batch_size, num_workers = num_workers)
