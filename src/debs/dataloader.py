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
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Custom Librairies
from dataset import BlackSea_Dataset


def time_encoding(time: torch.Tensor, frequencies:int = 128):
    r"""Encoding the time using the "Attention is all you need" paper encoding scheme"""

    # Security
    with torch.no_grad():

        # Encoding functions
        sinusoidal   = lambda time, frequency_index, frequencies: torch.sin(time / (10000 ** (frequency_index / frequencies)))
        cosinusoidal = lambda time, frequency_index, frequencies: torch.cos(time / (10000 ** (frequency_index / frequencies)))

        # Storing the encoding
        encoded_time = torch.zeros(time.shape[0], time.shape[1], frequencies * 2)

        # Mapping time to its encoding
        for b_index, b in enumerate(time):
            for t_index, t in enumerate(b):

                # Stores the current encoding
                encoding = list()

                # Computing the encoding, i.e. alternating between sinusoidal and cosinusoidal encoding
                for i in range(frequencies):
                    encoding += [sinusoidal(t, i, frequencies), cosinusoidal(t, i, frequencies)]

                # Conversion to torch tensor and storing the encoding
                encoded_time[b_index, t_index, :] =  torch.FloatTensor(encoding).clone()

        return encoded_time

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
                   bathymetry: np.array = None):

      # Extracting the data
      temperature = dataset.get_data("temperature")
      salinity    = dataset.get_data("salinity")
      chlorophyll = dataset.get_data("chlorophyll")
      height      = dataset.get_data("height")
      oxygen      = dataset.get_data("oxygen")

      # Creation of the input/output data
      self.x, self.y = np.stack([temperature, salinity, chlorophyll, height], axis = 1), oxygen

      # Creating time tensor (sample, (day, month, year)), encoding and concatenating embeddings
      self.encoded_time = torch.from_numpy(np.stack(dataset.get_time(), axis = 1))
      self.encoded_time = time_encoding(self.encoded_time, frequencies = frequencies)
      self.encoded_time = self.encoded_time.reshape(oxygen.shape[0], -1)

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
                         bathymetry: np.array,
                       window_input: int,
                      window_output: int,
                nb_samples_buffered: int):

                  # Initialization
                  self.x                   = x
                  self.y                   = y
                  self.t                   = t
                  self.mesh                = mesh
                  self.mask                = mask
                  self.bathymetry          = bathymetry
                  self.window_input        = window_input
                  self.window_output       = window_output
                  self.x_res               = self.x.shape[2]
                  self.y_res               = self.x.shape[3]
                  self.nb_buffered_samples = nb_samples_buffered

               def process(self, index : int):
                  """Used as a processing pipeline, i.e. it fetch and process a single data sample"""

                  # Determination of the indexes to perform the extraction
                  index_t        = self.nb_buffered_samples + index + 1
                  index_t_before = self.nb_buffered_samples + index + 1 - self.window_input
                  index_t_after  = self.nb_buffered_samples + index     + self.window_output

                  # Extracting the input and output
                  x = self.x[index_t_before : index_t]
                  y = self.y[index_t - 1    : index_t_after]

                  # Merging window intputs with physical variables values
                  x = x.reshape((-1, self.x_res, self.y_res))

                  # Masking the useless information (maybe needs to be changed)
                  x = np.where(self.mask == 0, 0, x)

                  # Stacking all the information
                  x = np.concatenate([x, self.mesh, self.bathymetry], axis = 0)

                  # Time
                  t = self.t[index_t]

                  # Returning the preprocessed samples (format changed for pytorch)
                  return x.astype(np.float32), t, y.astype(np.float32)

               def __len__(self):
                  return self.x.shape[0] - self.nb_buffered_samples - self.window_output

               def __getitem__(self, idx):
                  return self.process(index = idx)

         # Creation of the dataset for dataloader
         dataset = BS_Dataset(x = self.x,
                              y = self.y,
                              t = self.encoded_time,
                           mesh = self.mesh,
                           mask = self.mask,
                     bathymetry = self.bathymetry,
                   window_input = self.window_input,
                  window_output = self.window_output,
            nb_samples_buffered = self.nb_buffered_samples)

         # Creation of the dataloader
         return DataLoader(dataset, batch_size = self.batch_size, num_workers = self.num_workers)