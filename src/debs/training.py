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
import wandb
import time
import xarray
import numpy as np

# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim

# Custom libraries
from tools                  import *
from losses                 import loss_regression
from dataset                import BlackSea_Dataset
from metrics                import BlackSea_Metrics
from dataloader             import BlackSea_Dataloader
from neural_networks.loader import load_neural_network


def training(**kwargs):
  """Used to train a neural network to forecast the oxygen concentration in the Black Sea"""

  # -------------—---------
  #     Initialization
  # -------------—---------
  #
  # WandB
  project          = kwargs['Project']

  # Data
  start_month      = kwargs['Month (Starting)']
  end_month        = kwargs['Month (Ending)']
  start_year       = kwargs['Year (Starting)']
  end_year         = kwargs['Year (Ending)']
  inputs           = kwargs['Inputs']
  windows_inputs   = kwargs['Window (Inputs)']

  # Model
  architecture     = kwargs['Architecture']
  datasets_size    = kwargs['Datasets Size']
  learning_rate    = kwargs['Learning Rate']
  batch_size       = kwargs['Batch Size']
  nb_epochs        = kwargs['Epochs']

  # Loading the dataset helper tool
  BS_dataset = BlackSea_Dataset(year_start  = start_year,
                                year_end    = end_year,
                                month_start = start_month,
                                month_end   = end_month)

  # Loading the inputs
  input_datasets = [BS_dataset.get_data(variable = v) for v in inputs]

  # Loading the output
  data_oxygen = BS_dataset.get_data(variable = "oxygen")

  # Retrieving dimensions of the data
  timesteps, x_res, y_res = data_oxygen.shape

  # Loading the days, i.e. a tensor containing the ID of the days loaded to construct the dataset
  days_ID = BS_dataset.get_days()

  # Loading spatial information, i.e. the bathymetry (depth of the region) and the mesh (x, y coordinates)
  bathy = BS_dataset.get_depth(unit = "meter")
  mesh  = BS_dataset.get_mesh(x = x_res, y = y_res)

  # Hypoxia treshold
  hypox_tresh = xarray.open_dataset(BS_dataset.paths[0])["HYPON"].data.item()

  # Loading the different masks
  bs_mask             = BS_dataset.get_mask(continental_shelf = False)
  bs_mask_with_depth  = BS_dataset.get_mask(continental_shelf = True)
  bs_mask_complete    = get_complete_mask(data_oxygen, hypox_tresh, bs_mask_with_depth)

  # Retrieves the ratios of the different classes (Used to get insights about the data)
  ratio_oxygenated, ratio_switching, ratio_hypoxia = get_ratios(bs_mask_complete)

  # -------------—---------
  #     Preprocessing
  # -------------—---------
  #
  # Creating the dataloaders
  BS_loader = BlackSea_Dataloader(x = input_datasets,
                                  y = data_oxygen,
                                  t = days_ID,
                               mesh = mesh,
                               mask = bs_mask,
                         bathymetry = bathy,
                         window_inp = windows_inputs)

  dataset_train      = BS_loader.get_dataloader(type = "train",      batch_size = batch_size)
  dataset_validation = BS_loader.get_dataloader(type = "validation", batch_size = batch_size)
  dataset_test       = BS_loader.get_dataloader(type = "test",       batch_size = batch_size)

  # Extracting the number of samples in the validation set
  number_samples_validation = BS_loader.get_number_of_samples(type = "validation")

  # Generate random samplex indexes for the regression plot comparison
  random_samples_index = np.random.randint(0, number_samples_validation - 1, 10)

  # -------------—---------
  #        Settings
  # -------------—---------
  #
  # Fixing random seed for reproducibility
  np.random.seed(2701)

  # Information over terminal (1)
  project_title(kwargs)

  # Check if a GPU is available
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # -------------—---------
  #       Training
  # -------------—---------
  #
  # Loading the neural network
  neural_net = load_neural_network(architecture = architecture,
                                    data_output = data_oxygen,
                                         device = device,
                                         kwargs = kwargs)

  # Information over terminal (0)
  print("Number of trainable parameters: ", neural_net.count_parameters())

  # Pushing the neural network to the correct device
  neural_net.to(device)

  # Loading the optimizer
  optimizer  = optim.Adam(neural_net.parameters(), lr = learning_rate)

  # Used to compute the total time left,
  epoch_time = 0.0

  # WandB (1) - Initialization
  #wandb.init(project = project, mode = "disabled", config = kwargs)
  wandb.init(project = project, config = kwargs)

  # WandB (2) - Sending information about the datasets
  wandb.log({"Dataset & Architecture/Dataset (Visualisation, Image, Regions)" : wandb.Image(get_complete_mask_plot(bs_mask_complete)),
             "Dataset & Architecture/Dataset (Visualisation, Image, Ratios)"  : wandb.Image(get_ratios_plot(data_oxygen, hypox_tresh, bs_mask_with_depth)),
             "Dataset & Architecture/Ratio Oxygenated"                        : ratio_oxygenated,
             "Dataset & Architecture/Ratio Switching"                         : ratio_switching,
             "Dataset & Architecture/Ratio Hypoxia"                           : ratio_hypoxia,
             "Dataset & Architecture/Trainable Parameters"                    : neural_net.count_parameters()})

  # Main training loop
  for epoch in range(nb_epochs):

    # Timing the epoch
    start = time.time()

    # Used to store and compute instantaneous training loss
    training_loss, training_batch_steps = 0.0, 0

    # Used to compute our metrics and visual inspection for the validation set
    metrics_tool = BlackSea_Metrics(mask = bs_mask_with_depth,
                           mask_complete = bs_mask_complete,
                                treshold = hypox_tresh,
                       number_of_samples = number_samples_validation)

    # Stores predictions and validation samples (needed for pixelwise metrics)
    prediction_all, validation_all = None, None

    # Training the neural network
    for x, y in dataset_train:

        # Pushing the data to the correct device
        x, y = x.to(device), y.to(device)

        # Prediction of the neural network
        prediction = neural_net.forward(x)

        # Computing the loss
        loss_training = loss_regression(y_pred = prediction, y_true = y, mask = bs_mask_with_depth)

        # Information over terminal (2)
        progression(epoch = epoch,
             number_epoch = nb_epochs,
            loss_training = loss_training.item(),
          loss_validation = 0,
        loss_training_aob = 0,
      loss_validation_aob = 0)

        # WandB (3) - Sending information about the training loss
        wandb.log({f"Training/Loss (T)": loss_training.item()})

        # Accumulating the loss and updating the number of steps
        training_loss        += loss_training.item()
        training_batch_steps += 1

        # AverageNet - No optimization needed !
        if architecture == "AVERAGE":
            continue

        # Reseting the gradients
        optimizer.zero_grad()

        # Backward pass
        loss_training.backward()

        # Optimizing the parameters
        optimizer.step()

        # Cleaning
        x, y, prediction = x.to("cpu"), y.to("cpu"), prediction.to("cpu")
        del x, y, prediction, loss_training

    # Information over terminal (3)
    progression(epoch = epoch,
         number_epoch = nb_epochs,
        loss_training = training_loss / training_batch_steps,
    loss_training_aob = training_loss / training_batch_steps,
      loss_validation = 0,
  loss_validation_aob = 0)

    # WandB (4) - Sending information about the training loss
    wandb.log({f"Training/Loss (Training): ": training_loss / training_batch_steps})

    with torch.no_grad():

        # Used to store and compute instantaneous validation loss
        validation_loss, validation_batch_steps = 0.0, 0

        # Validating the neural network
        for x, y in dataset_validation:

            # Pushing the data to the correct device
            x, y = x.to(device), y.to(device)

            # Prediction of the neural network
            prediction = neural_net.forward(x)

            # Computing the loss
            loss_validation = loss_regression(y_pred = prediction, y_true = y, mask = bs_mask_with_depth)

            # Information over terminal (4)
            progression(epoch = epoch,
                 number_epoch = nb_epochs,
                loss_training = training_loss / training_batch_steps,
            loss_training_aob = training_loss / training_batch_steps,
              loss_validation = loss_validation.item(),
          loss_validation_aob = 0)

            # WandB (5) - Sending information about the validation loss
            wandb.log({f"Training/Loss (V)": loss_validation.item()})

            # Accumulating the loss and updating the number of steps
            validation_loss        += loss_validation.item()
            validation_batch_steps += 1

            # Pushing everything back to the CPU
            x, y, prediction = x.to("cpu"), y.to("cpu"), prediction.to("cpu")

            # Accumulating the predictions
            prediction_all = torch.cat((prediction_all, prediction), dim = 0) if prediction_all is not None else prediction
            validation_all = torch.cat((validation_all, y),          dim = 0) if validation_all is not None else y

            # Cleaning
            del x, y, prediction, loss_validation
            torch.cuda.empty_cache()

        # Metrics - Computing all the different metrics
        metrics_tool.compute_metrics(y_pred = prediction_all,
                                     y_true = validation_all)

        # Visualization - Comparaison plot
        cmp_plots = [metrics_tool.compute_plots_comparison_regression(y_pred = prediction_all,
                                                                      y_true = validation_all,
                                                                      index  = i) for i in random_samples_index]

        # Visualization - Pixelwise metrics
        metrics_tool.compute_plots(y_pred = prediction_all,
                                   y_true = validation_all)


        # Visualization - Global AUC
        fp, tp, auc, auc_plot = metrics_tool.compute_plot_ROCAUC_global(y_pred = prediction_all,
                                                                        y_true = validation_all,
                                                          normalized_threshold = hypox_tresh)

        # Information over terminal (5)
        progression(epoch = epoch,
             number_epoch = nb_epochs,
            loss_training = training_loss / training_batch_steps,
        loss_training_aob = training_loss / training_batch_steps,
          loss_validation = validation_loss / validation_batch_steps,
      loss_validation_aob = validation_loss / validation_batch_steps)

        # Updating timing
        epoch_time = time.time() - start

        # Getting results of each metric (averaged over each batch)
        results, results_name = metrics_tool.get_results()

        # Getting the plots of each metric
        plots, plots_name = metrics_tool.get_plots()

        # WandB (6) - Sending validation loss and visual information
        wandb.log({"Training/Loss (Validation)"                             : validation_loss / validation_batch_steps,
                   "Training/Epochs"                                        : nb_epochs - epoch,
                   "Training/Time Left"                                     : (nb_epochs - epoch) * epoch_time,
                   f"Metrics/Area Under The Curve (Global)"                 : auc,
                   f"Visualization (Metrics)/Area Under The Curve (Global)" : wandb.Image(auc_plot)})

        # WandB (7) - Sending visual information
        for i, p in enumerate(cmp_plots):
            wandb.log({f"Visualization (Prediction VS Ground Truth)/Sample {i}" : wandb.Image(p)})

        # WandB (8) - Sending metrics scores
        for d, day_results in enumerate(results):
            for i, result in enumerate(day_results):
              wandb.log({f"Metrics/{results_name[i]}" : result})

        # WandB (8) - Sending visual information
        for plot, name in zip(plots, plots_name):
          wandb.log({f"Visualization (Metrics)/{name}" : wandb.Image(plot)})

    # Clearing the plot
    plt.clf()
    plt.close()

  # Extracting the Neural Network back to CPU
  neural_net.to("cpu")

  # Finishing the Weight and Biases run
  wandb.finish()
