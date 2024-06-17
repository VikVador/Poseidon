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
# Function used to train a neural network to forecast the oxygen concentration in the Black Sea.
#
import time
import wandb
import xarray
import calendar
import numpy as np

# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR

# Custom libraries
from metrics                import *
from tools                  import *
from losses                 import *
from dataset                import BlackSea_Dataset
from dataloader             import BlackSea_Dataloader
from neural_networks.loader import load_neural_network


def training(**kwargs):
  """Used to train a neural network to forecast the oxygen concentration in the Black Sea"""

  # -------------—---------
  #     Initialization
  # -------------—---------
  #
  # Checking if GPU is available
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Fixing random seed for reproducibility
  np.random.seed(2701)
  torch.manual_seed(2701)

  # Loading configuration
  project        = kwargs['Project']
  mode           = kwargs['Mode']
  window_input   = kwargs['Window (Inputs)']
  window_output  = kwargs['Window (Outputs)']
  frequencies    = kwargs['Frequencies']
  architecture   = kwargs['Architecture']
  learning_rate  = kwargs['Learning Rate']
  batch_size     = kwargs['Batch Size']
  nb_epochs      = kwargs['Epochs']
  num_workers    = kwargs['Number of Workers']

  # -------------—---------
  #     Loading the data
  # -------------—---------
  # Displaying information over terminal (1)
  print("Loading the data...")

  training   = BlackSea_Dataset("Training")
  validation = BlackSea_Dataset("Validation")
  test       = BlackSea_Dataset("Test")

  # Extracting the output (used by AverageNET)
  data_oxygen = training.get_data(variable = "oxygen")

  # Loading the mesh, masks and bathymetry
  mesh                = test.get_mesh()
  bs_mask             = test.get_mask(continental_shelf = False)
  bs_mask_with_depth  = test.get_mask(continental_shelf = True)
  bathymetry          = test.get_depth(unit = "meter")

  # Hypoxia treshold
  hypox_tresh = training.get_treshold(standardized = True)

  # -------------—---------
  #     Preprocessing
  # -------------—---------
  # Displaying information over terminal (2)
  print("Preprocessing the data...")

  BS_loader_train = BlackSea_Dataloader(training,
                                        window_input,
                                        window_output,
                                        frequencies,
                                        batch_size,
                                        num_workers,
                                        mesh,
                                        bs_mask,
                                        bs_mask_with_depth,
                                        bathymetry,
                                        random = True)

  BS_loader_validation = BlackSea_Dataloader(validation,
                                             window_input,
                                             window_output,
                                             frequencies,
                                             batch_size,
                                             num_workers,
                                             mesh,
                                             bs_mask,
                                             bs_mask_with_depth,
                                             bathymetry,
                                             random = False)

  BS_loader_test = BlackSea_Dataloader(test,
                                       window_input,
                                       window_output,
                                       frequencies,
                                       batch_size,
                                       num_workers,
                                       mesh,
                                       bs_mask,
                                       bs_mask_with_depth,
                                       bathymetry,
                                       random = False)


  # Creating the dataloaders
  dataset_train        = BS_loader_train.get_dataloader()
  dataset_validation   = BS_loader_validation.get_dataloader()
  dataset_test         = BS_loader_test.get_dataloader()

  # -------------—--------------------
  #     Neural Network & Training
  # -------------—--------------------
  # Displaying information over terminal (2)
  print("Starting to train...")

  # Initialization of the neural network
  neural_net = load_neural_network(architecture = architecture, data_output = None, device = device, kwargs = kwargs)
  neural_net.to(device)

  # Total number of available GPUs
  num_gpus = torch.cuda.device_count()
  print("Available GPUs: ", num_gpus)

  # Total number of parameters
  nn_params = neural_net.count_parameters()

  # Using multiple GPUS
  neural_net = torch.nn.parallel.DataParallel(neural_net, device_ids=list(range(num_gpus)), dim = 0)

  # Loading the optimizer
  optimizer = optim.Adam(neural_net.parameters(), lr = learning_rate)

  # Loading the scheduler
  scheduler = LinearLR(optimizer, start_factor = 0.95, total_iters = nb_epochs)

  # WandB (1) - Initialization of the run
  wandb.init(project = project, mode = mode, config = kwargs)
  wandb.log({f"Total number of parameters": nn_params})

  # Used to compute the total time left,
  epoch_time = 0.0

  # ------- Training Loop -------
  for epoch in range(nb_epochs):

    # Timing the epoch
    start = time.time()

    # Used to store and compute instantaneous training loss
    loss_training_total, loss_training_per_day, loss_training_index = 0.0, list(), 0

    # Used to compute metrics
    metrics = BlackSea_Metrics(data_oxygen = data_oxygen, mask = bs_mask_with_depth, hypoxia_treshold = hypox_tresh, window_output = window_output, number_trajectories = 25)

    # Training the neural network
    for x, t, y in dataset_train:

      # Pushing the data to the correct device
      x, t, y = x.to(device), t.to(device), y.to(device)

      # Forward pass
      pred = neural_net.forward(x, t)

      # Computing the training loss
      loss_training_batch_total, loss_training_batch_per_day = forecasting_loss(y_true = y,
                                                                                y_pred = pred,
                                                                                  mask = bs_mask_with_depth)

      # Accumulating the total loss, storing losses per day and updating the number of training steps
      loss_training_total += loss_training_batch_total.item()
      loss_training_index += 1
      loss_training_per_day.append([l.item() for l in loss_training_batch_per_day])

      # AverageNet : No optimization needed !
      if architecture == "AVERAGE":
          continue

      # Reseting the gradients
      optimizer.zero_grad()

      # Backward pass
      loss_training_batch_total.backward()

      # Optimizing the parameters
      optimizer.step()

      # Freeing the GPU
      x, y, t, pred = x.to("cpu"), y.to("cpu"), t.to("cpu"), pred.to("cpu")

      # WandB (2.1) - Sending information about the training results
      wandb.log({f"Training/Loss (T)": loss_training_batch_total.item()})
      wandb.log({f"Training/Loss (T, {i})": loss.item() for i, loss in enumerate(loss_training_batch_per_day)})

      # Freeing memory
      del x, t, y, pred, loss_training_batch_total, loss_training_batch_per_day

    # WandB (2.2) - Sending information about the training results
    wandb.log({f"Training/Loss (Training): ": loss_training_total / loss_training_index})

    with torch.no_grad():

      # Used to store and compute instantaneous training loss
      loss_validation_total, loss_validation_per_day, loss_validation_index = 0.0, list(), 0

      # Validating the neural network
      for x, t, y in dataset_validation:

        # Pushing the data to the correct device
        x, t, y = x.to(device), t.to(device), y.to(device)

        # Forward pass
        pred = neural_net.forward(x, t)

        # Computing the validation loss
        loss_validation_batch_total, loss_validation_batch_per_day = forecasting_loss(y_true = y,
                                                                                      y_pred = pred,
                                                                                        mask = bs_mask_with_depth)

        # Accumulating the total loss, storing losses per day and updating the number of training steps
        loss_validation_total += loss_validation_batch_total.item()
        loss_validation_index += 1
        loss_validation_per_day.append([l.item() for l in loss_validation_batch_per_day])

        # Pushing everything back to the CPU
        x, y, t, pred = x.to("cpu"), y.to("cpu"), t.to("cpu"), pred.to("cpu")

        # WandB (3.1) - Sending information about the validation results
        wandb.log({f"Training/Loss (V)": loss_validation_batch_total.item()})
        wandb.log({f"Training/Loss (V, {i})": loss.item() for i, loss in enumerate(loss_validation_batch_per_day)})

        # Computing metrics
        metrics.analyze(pred, y)

        # Cleaning
        del x, t, y, pred, loss_validation_batch_total, loss_validation_batch_per_day
        torch.cuda.empty_cache()

    # Displaying information over terminal
    print(f"Epoch {epoch + 1}/{nb_epochs} | \
            Training = {(loss_training_total / loss_training_index)} | \
            Validation = {(loss_validation_total / loss_validation_index)}")

    # Updating the scheduler
    scheduler.step()

    # Timing the epoch (2)
    epoch_time = (time.time() - start)/3600

    # WandB (2.2) - Sending information about the training results
    wandb.log({f"Training/Loss (Validation): ": loss_validation_total / loss_validation_index,
               f"Training/Time Left (H)": epoch_time})

    # WandB (2.3) - Computing and sending the results
    metrics.send_results()

  # Extracting the Neural Network back to CPU
  neural_net.to("cpu")

  # Finishing the Weight and Biases run
  wandb.finish()
