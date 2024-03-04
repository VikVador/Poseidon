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
# A (main) function used to train a neural network to forecast the oxygen concentration in the Black Sea.
#
import wandb
import time

# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim

# Custom libraries
from tools                  import *
from losses                 import compute_loss
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
    problem          = kwargs['Problem']
    hypoxia_treshold = kwargs['Hypoxia Treshold']
    depth            = kwargs['Depth']
    inputs           = kwargs['Inputs']
    windows_inputs   = kwargs['Window (Inputs)']
    windows_outputs  = kwargs['Window (Output)']
    windows_transfo  = kwargs['Window (Transformation)']

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
    input_datasets = [BS_dataset.get_data(variable = v, region = "surface") for v in inputs]

    # Loading the output
    data_oxygen = BS_dataset.get_data(variable = "oxygen", region = "bottom", depth = depth)

    # Retrieving dimensions of the data
    timesteps, x_res, y_res = data_oxygen.shape

    # Loading the days, i.e. a tensor containing the ID of the days loaded to construct the dataset
    days_ID = BS_dataset.get_days()

    # Loading spatial information, i.e. the bathymetry (depth of the region) and the mesh (x, y coordinates)
    bathy = BS_dataset.get_depth(unit = "meter")
    mesh  = BS_dataset.get_mesh(x = x_res, y = y_res)

    # Loading the different masks
    bs_mask             = BS_dataset.get_mask(depth = None)
    bs_mask_with_depth  = BS_dataset.get_mask(depth = depth)
    bs_mask_complete    = get_complete_mask(data_oxygen, bs_mask_with_depth)

    # -------------—---------
    #     Preprocessing
    # -------------—---------
    #
    # Cropping dimensions to be a multiple of 2
    for i, v in enumerate(input_datasets):
        input_datasets[i] = crop(v, factor = 2)

    data_oxygen        = crop(data_oxygen,        factor = 2)
    mesh               = crop(mesh,               factor = 2)
    bathy              = crop(bathy,              factor = 2)
    bs_mask            = crop(bs_mask,            factor = 2)
    bs_mask_with_depth = crop(bs_mask_with_depth, factor = 2)
    bs_mask_complete   = crop(bs_mask_complete,   factor = 2)

    # Creating the dataloader
    BS_loader = BlackSea_Dataloader(x = input_datasets,
                                    y = data_oxygen,
                                    t = days_ID,
                                 mesh = mesh,
                                 mask = bs_mask,
                      mask_with_depth = bs_mask_with_depth,
                           bathymetry = bathy,
                           window_inp = windows_inputs,
                           window_out = windows_outputs,
                       window_transfo = windows_transfo,
                                 mode = problem,
                     hypoxia_treshold = hypoxia_treshold,
                        datasets_size = datasets_size)

    # Preprocessing the data
    dataset_train      = BS_loader.get_dataloader(type = "train",      batch_size = batch_size)
    dataset_validation = BS_loader.get_dataloader(type = "validation", batch_size = batch_size)
    dataset_test       = BS_loader.get_dataloader(type = "test",       batch_size = batch_size)

    # Extracting the normalized oxygen treshold value
    norm_oxy = BS_loader.get_normalized_deoxygenation_treshold()

    # Extracting the number of samples in the validation set
    number_samples_validation = BS_loader.get_number_of_samples(type = "validation")

    # Retrieves the ratios of the different classes (Used to get insights about the data)
    ratio_oxygenated, ratio_switching, ratio_hypoxia = get_ratios(bs_mask_complete)

    # -------------—---------
    #        Settings
    # -------------—---------
    #
    # Check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading the neural network
    neural_net = load_neural_network(architecture = architecture,
                                      data_output = data_oxygen,
                                           device = device,
                                           kwargs = kwargs)

    # Pushing the neural network to the correct device
    neural_net.to(device)

    # Loading the optimizer
    optimizer  = optim.Adam(neural_net.parameters(), lr = learning_rate)

    # Loading the loss information
    loss_type = "MSE" if problem == "regression" else "BCE"

    # Used to compute the total time left,
    epoch_time = 0.0

    # Information over terminal (1)
    project_title(kwargs)

    # WandB (1) - Initialization
    wandb.init(project = project, config = kwargs)

    # WandB (2) - Sending information about the datasets
    wandb.log({"Dataset & Architecture/Dataset (Visualisation, Image, Regions)" : wandb.Image(get_complete_mask_plot(bs_mask_complete)),
                "Dataset & Architecture/Dataset (Visualisation, Image, Ratios)" : wandb.Image(get_ratios_plot(data_oxygen, bs_mask_with_depth)),
                "Dataset & Architecture/Dataset (Visualisation, Video, Oxygen)" : wandb.Video(get_video(data = data_oxygen), fps = 1),
                "Dataset & Architecture/Ratio Oxygenated"                       : ratio_oxygenated,
                "Dataset & Architecture/Ratio Switching"                        : ratio_switching,
                "Dataset & Architecture/Ratio Hypoxia"                          : ratio_hypoxia,
                "Dataset & Architecture/Trainable Parameters"                   : neural_net.count_parameters()})

    # -------------—---------
    #        Training
    # -------------—---------
    for epoch in range(nb_epochs):

        # Timing the epoch
        start = time.time()

        # Used to store and compute instantaneous training loss
        training_loss, training_batch_steps = 0.0, 0

        # Used to compute our metrics and visual inspection for the validation set
        metrics_tool = BlackSea_Metrics(mode = problem,
                                        mask = bs_mask_with_depth,
                               mask_complete = bs_mask_complete,
                                    treshold = norm_oxy,
                           number_of_samples = number_samples_validation)


        # Training the neural network
        for x, t, y in dataset_train:

            # Pushing the data to the correct device
            x, t, y = x.to(device), t.to(device), y.to(device)

            # Prediction of the neural network
            prediction = neural_net.forward(x, t)

            # Computing the loss
            loss_training = compute_loss(y_pred = prediction,
                                         y_true = y,
                                        problem = problem,
                                         device = device,
                                         kwargs = kwargs)

            # Information over terminal (2)
            progression(epoch = epoch,
                 number_epoch = nb_epochs,
                loss_training = loss_training.detach().item(),
              loss_validation = 0,
            loss_training_aob = 0,
          loss_validation_aob = 0)

            # WandB (3) - Sending information about the training loss
            wandb.log({f"Training/Loss ({loss_type}, T)": loss_training.detach().item()})

            # Accumulating the loss and updating the number of steps
            training_loss        += loss_training.detach().item()
            training_batch_steps += 1

            # AverageNet - No optimization needed !
            if architecture == "AVERAGE": continue

            # Reseting the gradients
            optimizer.zero_grad()

            # Backward pass
            loss_training.backward()

            # Optimizing the parameters
            optimizer.step()

            break

        # Information over terminal (3)
        progression(epoch = epoch,
             number_epoch = nb_epochs,
            loss_training = loss_training.detach().item(),
        loss_training_aob = training_loss / training_batch_steps,
          loss_validation = 0,
      loss_validation_aob = 0)

        # WandB (4) - Sending information about the training loss
        wandb.log({f"Training/Loss ({loss_type}, Training): ": training_loss / training_batch_steps})

        with torch.no_grad():

            # Used to store and compute instantaneous validation loss
            validation_loss, validation_batch_steps = 0.0, 0

            # Validating the neural network
            for x, t, y in dataset_validation:

                # Pushing the data to the correct device
                x, t, y = x.to(device), t.to(device), y.to(device)

                # Prediction of the neural network
                prediction = neural_net.forward(x, t)

                # Computing the loss
                loss_validation = compute_loss(y_pred = prediction,
                                               y_true = y,
                                              problem = problem,
                                               device = device,
                                               kwargs = kwargs)

                # Information over terminal (4)
                progression(epoch = epoch,
                     number_epoch = nb_epochs,
                    loss_training = loss_training.detach().item(),
                loss_training_aob = training_loss / training_batch_steps,
                  loss_validation = loss_validation.detach().item(),
              loss_validation_aob = 0)

                # WandB (4) - Sending information about the validation loss
                wandb.log({f"Training/Loss ({loss_type}, V)": loss_validation.detach().item()})

                # Accumulating the loss and updating the number of steps
                validation_loss        += loss_validation.detach().item()
                validation_batch_steps += 1

                # Transforming to probabilities (not done in forward pass because BCEWithLogitsLoss does it for us)
                x = nn.Softmax(dim = 2)(prediction) if problem == "classification" else prediction

                # Pushing the data to te CPU (needed to compute metrics)
                prediction, y = prediction.to("cpu"), y.to("cpu")

                # Used to compute the metrics
                metrics_tool.compute_metrics(y_pred = torch.unsqueeze(prediction[:, 0], dim = 1), y_true = y)
                metrics_tool.compute_plots(  y_pred = torch.unsqueeze(prediction[:, 0], dim = 1), y_true = y) if validation_batch_steps == 1 else None

                # WandB (5) - Sending visual information about the validation
                if problem == "regression":
                  wandb.log({f"Visualization/Prediction VS Ground Truth (Regression)": wandb.Image(metrics_tool.compute_plots_comparison_regression(y_pred = prediction, y_true = y))}) if validation_batch_steps == 1 else None
                else:
                  wandb.log({f"Visualization/Prediction VS Ground Truth (Classification)": wandb.Image(metrics_tool.compute_plots_comparison_classification(y_pred = prediction, y_true = y))}) if validation_batch_steps == 1 else None

                break

            # Information over terminal (5)
            progression(epoch = epoch,
                 number_epoch = nb_epochs,
                loss_training = loss_training.detach().item(),
            loss_training_aob = training_loss / training_batch_steps,
              loss_validation = loss_validation.detach().item(),
          loss_validation_aob = validation_loss / validation_batch_steps)

            # Updating timing
            epoch_time = time.time() - start

            # Getting results of each metric (averaged over each batch)
            results, results_name = metrics_tool.get_results()

            # Getting the plots of each metric
            plots, plots_name = metrics_tool.get_plots()

            # WandB (6) - Sending information about the validation loss
            wandb.log({f"Training/Loss ({loss_type}, Validation)": validation_loss / validation_batch_steps,
                        "Training/Epochs"                        : nb_epochs - epoch,
                        "Training/Time Left"                     : (nb_epochs - epoch) * epoch_time})

            # WandB (7) - Sending metrics scores
            for d, day_results in enumerate(results):
                for i, result in enumerate(day_results):

                  # Metric with corresponding forecasted day (Only if more than 1 day is forecasted)
                  m_name = results_name[i] + " D(" + str(d) + ")" if windows_outputs > 1 else results_name[i]

                  # Logging
                  wandb.log({f"Metrics/{m_name}" : result})

            # WandB (8) - Sending visual information
            for plot, name in zip(plots, plots_name):

              # Logging
              wandb.log({f"Visualization/{name}" : wandb.Image(plot)})

    # Finishing the Weight and Biases run
    wandb.finish()