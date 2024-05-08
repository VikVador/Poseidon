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
# A tool to create a dataloader that processes data on the fly
#
import numpy as np

# Torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Custom Librairies
from dataset import BlackSea_Dataset


class BlackSea_Dataloader():
   r"""A tool to create a dataloader that processes and loads the Black Sea datasets on the fly"""

   def __init__(self, dataset: BlackSea_Dataset,
                 window_input: int = 1,
                window_output: int = 10,
                   batch_size: int = 1,
                  num_workers: int = 2):

      # Extracting the data
      temperature = dataset.get_data("temperature")
      salinity    = dataset.get_data("salinity")
      chlorophyll = dataset.get_data("chlorophyll")
      height      = dataset.get_data("height")
      oxygen      = dataset.get_data("oxygen")

      # Creation of the input/output data
      self.x, self.y = np.stack([temperature, salinity, chlorophyll, height], axis = 1), oxygen

      # Extracting time information
      self.time_day, self.time_month, self.time_year = dataset.get_time()

      # Corresponds to the number of samples used as "buffer"
      self.nb_buffered_samples = 365

      # Storing other information
      self.window_input  = window_input
      self.window_output = window_output
      self.batch_size    = batch_size
      self.num_workers   = num_workers
      self.mesh          = dataset.get_mesh()
      self.mask          = dataset.get_mask(continental_shelf = False)
      self.mask_CS       = dataset.get_mask(continental_shelf = True)
      self.bathymetry    = dataset.get_depth(unit = "meter")

   def get_number_of_samples(self):
      r"""Returns the number of samples in the dataset"""
      return self.x.shape[0] - self.nb_buffered_samples - self.window_output

   def get_dataloader(self):
         r"""Creates and returns a dataloader"""

         class BS_Dataset(Dataset):
               r"""Pytorch dataloader"""

               def __init__(self, x: np.array,
                                  y: np.array,
                               mesh: np.array,
                               mask: np.array,
                           time_day: np.array,
                         time_month: np.array,
                          time_year: np.array,
                         bathymetry: np.array,
                       window_input: int,
                      window_output: int,
                nb_samples_buffered: int):

                  # Initialization
                  self.x                   = x
                  self.y                   = y
                  self.mesh                = mesh
                  self.mask                = mask
                  self.time_day            = time_day
                  self.time_month          = time_month
                  self.time_year           = time_year
                  self.bathymetry          = bathymetry
                  self.window_input        = window_input
                  self.window_output       = window_output
                  self.x_res               = self.x.shape[2]
                  self.y_res               = self.x.shape[3]
                  self.nb_buffered_samples = nb_samples_buffered
                  self.time_img            = np.ones((self.x_res, self.y_res))

               def process(self, index : int):
                  """Used as a processing pipeline, i.e. it fetch and process a single data sample"""

                  # Determination of the indexes to perform the extraction
                  index_t        = self.nb_buffered_samples + index + 1
                  index_t_before = self.nb_buffered_samples + index + 1 - self.window_input
                  index_t_after  = self.nb_buffered_samples + index     + self.window_output

                  # Extracting the input and output
                  x = self.x[index_t_before : index_t]
                  y = self.y[index_t - 1    : index_t_after]

                  # Creation of temporal data
                  t = np.stack([self.time_img * self.time_day[index_t],
                                self.time_img * self.time_month[index_t],
                                self.time_img * self.time_year[index_t]], axis = 0)

                  # Merging window intputs with physical variables values
                  x = x.reshape((-1, self.x_res, self.y_res))

                  # Masking the useless information (maybe needs to be changed)
                  x = np.where(self.mask == 0, 0, x)

                  # Stacking all the information
                  x = np.concatenate([x, self.mesh, self.bathymetry, t], axis = 0)

                  # Returning the preprocessed samples (format changed for pytorch)
                  return x.astype(np.float32), y.astype(np.float32)

               def __len__(self):
                  return self.x.shape[0] - self.nb_buffered_samples - self.window_output

               def __getitem__(self, idx):
                  return self.process(index = idx)

         # Creation of the dataset for dataloader
         dataset = BS_Dataset(x = self.x,
                              y = self.y,
                           mesh = self.mesh,
                           mask = self.mask,
                       time_day = self.time_day,
                     time_month = self.time_month,
                      time_year = self.time_year,
                     bathymetry = self.bathymetry,
                   window_input = self.window_input,
                  window_output = self.window_output,
            nb_samples_buffered = self.nb_buffered_samples)

         # Creation of the dataloader
         return DataLoader(dataset, batch_size = self.batch_size, num_workers = self.num_workers)