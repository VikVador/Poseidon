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
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Custom Librairies
from dataset import BlackSea_Dataset


class BlackSea_Dataloader():
   r"""A tool to create a dataloader that processes and loads the Black Sea datasets on the fly"""

   def __init__(self, dataset: BlackSea_Dataset,
                 window_input: int = 1,
                window_output: int = 10,
                  frequencies: int = 128,
                   batch_size: int = 1,
                  num_workers: int = 2,
                         mesh: np.array = None,
                         mask: np.array = None,
                      mask_CS: np.array = None,
                   bathymetry: np.array = None,
                       random: bool = False):

        # Target Variable
        oxygen = dataset.get_data("oxygen")

        # Used for conditioning the prediction (1)
        temperature = dataset.get_data("temperature")
        salinity    = dataset.get_data("salinity")
        chlorophyll = dataset.get_data("chlorophyll")
        height      = dataset.get_data("height")

        # Creation of the input/output data
        self.x, self.y = np.stack([temperature, salinity, chlorophyll, height], axis = 1), oxygen

        # Used for conditioning the prediction (2)
        self.t = torch.from_numpy(np.stack(dataset.get_time(), axis = 1))

        # Corresponds to the number of samples used as "buffer"
        self.nb_buffered_samples = 365

        # Storing other information
        self.window_input  = window_input
        self.window_output = window_output
        self.batch_size    = batch_size
        self.num_workers   = num_workers
        self.mesh          = mesh
        self.mask          = mask
        self.mask_CS       = mask_CS
        self.bathymetry    = bathymetry
        self.random        = random

   def get_number_of_samples(self):
      r"""Returns the number of samples in the dataset"""
      return self.x.shape[0] - self.nb_buffered_samples - self.window_output

   def get_dataloader(self):
         r"""Creates and returns a dataloader"""

         class BS_Dataset(Dataset):
               r"""Pytorch dataloader"""

               def __init__(self, x: np.array,
                                  y: np.array,
                                  t: np.array,
                               mesh: np.array,
                               mask: np.array,
                            mask_CS: np.array,
                         bathymetry: np.array,
                       window_input: int,
                      window_output: int,
                nb_samples_buffered: int,
                             random: bool):

                  # Initialization
                  self.x                   = x
                  self.y                   = y
                  self.t                   = t
                  self.mesh                = mesh
                  self.mask                = mask
                  self.mask_CS             = mask_CS
                  self.bathymetry          = bathymetry
                  self.window_input        = window_input
                  self.window_output       = window_output
                  self.x_res               = self.x.shape[2]
                  self.y_res               = self.x.shape[3]
                  self.nb_buffered_samples = nb_samples_buffered
                  self.random              = random
                  self.random_indexes      = np.random.permutation(np.indices(x[:(- self.nb_buffered_samples - self.window_output), 0, 0, 0].shape)[0])

               def process(self, index : int):
                  """Used as a processing pipeline, i.e. it fetch and process a single data sample"""

                  # Determination of the indexes to perform the extraction
                  index_t        = self.nb_buffered_samples + index + 1
                  index_t_before = self.nb_buffered_samples + index + 1 - self.window_input
                  index_t_after  = self.nb_buffered_samples + index     + self.window_output

                  # ---- Conditioning ----
                  #
                  # Extracting the surface variables conditionning the prediction
                  x = self.x[index_t_before : index_t]

                  # Merging window intputs with physical variables values
                  x = x.reshape((-1, self.x_res, self.y_res))

                  # Masking the useless information
                  x = np.where(self.mask == 0, 0, x)

                  # Stacking all the physical variables information
                  x = np.concatenate([x, self.mesh, self.bathymetry], axis = 0)

                  # Extracting the temporal conditionning
                  t = self.t[index_t]

                  # Stacking the time information
                  x = np.concatenate([x, t[:, None, None].expand(3, x.shape[1], x.shape[2])], axis = 0)

                  # ----- Target -----
                  #
                  # Extracting the target
                  y = self.y[index_t - 1 : index_t_after]

                  # Masking the useless information
                  y = np.where(self.mask_CS == 0, 0, y)

                  # Returning the preprocessed samples (format changed for pytorch)
                  return x.astype(np.float32), t, y.astype(np.float32)

               def __len__(self):
                  return self.x.shape[0] - self.nb_buffered_samples - self.window_output

               def __getitem__(self, idx):

                  # Random sampling
                  idx = idx if not self.random else self.random_indexes[idx]

                  # Processing the data
                  return self.process(index = idx)

         # Creation of the dataset for dataloader
         dataset = BS_Dataset(x = self.x,
                              y = self.y,
                              t = self.t,
                           mesh = self.mesh,
                           mask = self.mask,
                        mask_CS = self.mask_CS,
                     bathymetry = self.bathymetry,
                   window_input = self.window_input,
                  window_output = self.window_output,
            nb_samples_buffered = self.nb_buffered_samples,
                         random = self.random)

         # Creation of the dataloader
         return DataLoader(dataset, batch_size = self.batch_size, num_workers = self.num_workers)

class BlackSea_Dataloader_Diffusion():
   r"""A tool to create a dataloader that processes and loads the Black Sea datasets on the fly only for the first day of each month (used for validation)"""

   def __init__(self, dataset: BlackSea_Dataset,
                 window_input: int = 1,
                window_output: int = 10,
                  frequencies: int = 128,
                   batch_size: int = 1,
                  num_workers: int = 2,
                         mesh: np.array = None,
                         mask: np.array = None,
                      mask_CS: np.array = None,
                   bathymetry: np.array = None,
                       random: bool = False):

         # Target Variable
         oxygen = dataset.get_data("oxygen")

         # Used for conditioning the prediction (1)
         temperature = dataset.get_data("temperature")
         salinity    = dataset.get_data("salinity")
         chlorophyll = dataset.get_data("chlorophyll")
         height      = dataset.get_data("height")

         # Creation of the input/output data
         self.x, self.y = np.stack([temperature, salinity, chlorophyll, height], axis = 1), oxygen

         # Used for conditioning the prediction (2)
         self.t = torch.from_numpy(np.stack(dataset.get_time(), axis = 1))

         # Find the indices of each month
         self.transitions = torch.nonzero(self.t[:-1, 1] != self.t[1:, 1]).squeeze() + 1
         self.transitions = torch.cat((torch.tensor([0]), self.transitions))[12:] - 365

         # Corresponds to the number of samples used as "buffer"
         self.nb_buffered_samples = 365

         # Storing other information
         self.window_input  = window_input
         self.window_output = window_output
         self.batch_size    = batch_size
         self.num_workers   = num_workers
         self.mesh          = mesh
         self.mask          = mask
         self.mask_CS       = mask_CS
         self.bathymetry    = bathymetry
         self.random        = random

   def get_number_of_samples(self):
      r"""Returns the number of samples in the dataset"""
      return self.transitions.shape[0]

   def get_dataloader(self):
         r"""Creates and returns a dataloader"""

         class BS_Dataset(Dataset):
               r"""Pytorch dataloader"""

               def __init__(self, x: np.array,
                                  y: np.array,
                                  t: np.array,
                               mesh: np.array,
                               mask: np.array,
                            mask_CS: np.array,
                         bathymetry: np.array,
                       window_input: int,
                      window_output: int,
                nb_samples_buffered: int,
                      index_samples: torch.tensor,
                             random: bool):

                  # Initialization
                  self.x                   = x
                  self.y                   = y
                  self.t                   = t
                  self.idx                 = index_samples
                  self.mesh                = mesh
                  self.mask                = mask
                  self.mask_CS             = mask_CS
                  self.bathymetry          = bathymetry
                  self.window_input        = window_input
                  self.window_output       = window_output
                  self.x_res               = self.x.shape[2]
                  self.y_res               = self.x.shape[3]
                  self.nb_buffered_samples = nb_samples_buffered
                  self.random              = random
                  self.random_indexes      = np.random.permutation(np.indices(x[:(- self.nb_buffered_samples - self.window_output), 0, 0, 0].shape)[0])

               def process(self, index : int):
                  """Used as a processing pipeline, i.e. it fetch and process a single data sample"""

                  # Determination of the indexes to perform the extraction
                  index_t        = self.nb_buffered_samples + index + 1
                  index_t_before = self.nb_buffered_samples + index + 1 - self.window_input
                  index_t_after  = self.nb_buffered_samples + index     + self.window_output

                  # ---- Conditioning ----
                  #
                  # Extracting the surface variables conditionning the prediction
                  x = self.x[index_t_before : index_t]

                  # Merging window intputs with physical variables values
                  x = x.reshape((-1, self.x_res, self.y_res))

                  # Masking the useless information
                  x = np.where(self.mask == 0, 0, x)

                  # Stacking all the physical variables information
                  x = np.concatenate([x, self.mesh, self.bathymetry], axis = 0)

                  # Extracting the temporal conditionning
                  t = self.t[index_t]

                  # Stacking the time information
                  x = np.concatenate([x, t[:, None, None].expand(3, x.shape[1], x.shape[2])], axis = 0)

                  # ----- Target -----
                  #
                  # Extracting the target
                  y = self.y[index_t - 1 : index_t_after]

                  # Masking the useless information
                  y = np.where(self.mask_CS == 0, 0, y)

                  # Returning the preprocessed samples (format changed for pytorch)
                  return x.astype(np.float32), t, y.astype(np.float32)

               def __len__(self):
                  return self.idx.shape[0]

               def __getitem__(self, idx):
                  return self.process(index = self.idx[idx])

         # Creation of the dataset for dataloader
         dataset = BS_Dataset(x = self.x,
                              y = self.y,
                              t = self.t,
                           mesh = self.mesh,
                           mask = self.mask,
                        mask_CS = self.mask_CS,
                     bathymetry = self.bathymetry,
                   window_input = self.window_input,
                  window_output = self.window_output,
            nb_samples_buffered = self.nb_buffered_samples,
                  index_samples = self.transitions,
                         random = self.random)

         # Creation of the dataloader
         return DataLoader(dataset, batch_size = self.batch_size, num_workers = self.num_workers)