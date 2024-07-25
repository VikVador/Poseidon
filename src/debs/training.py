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
import tqdm
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
from unet                   import *
from dataset                import BlackSea_Dataset
from dataloader             import BlackSea_Dataloader, BlackSea_Dataloader_Diffusion


def training(**kwargs):
    """Used to train a neural network to forecast the oxygen concentration in the Black Sea"""

    # -------------—---------
    #     Initialization
    # -------------—---------
    #
    # Information over terminal (1)
    project_title(kwargs)

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
    diff_steps     = kwargs['Diffusion Steps']
    diff_scheduler = kwargs['Diffusion Scheduler']
    diff_variance  = kwargs['Diffusion Variance']
    scaling        = kwargs['Scaling']
    frequencies    = kwargs['Frequencies']
    learning_rate  = kwargs['Learning Rate']
    batch_size     = kwargs['Batch Size']
    nb_epochs      = kwargs['Epochs']
    num_workers    = kwargs['Number of Workers']

    # -------------—---------
    #          Data
    # -----------------------
    #
    # Loading preprocessed datasets
    dataset_train      = BlackSea_Dataset("Validation")
    dataset_validation = BlackSea_Dataset("Test")

    # Loading other information
    black_sea_mesh       = dataset_validation.get_mesh()
    black_sea_mask       = dataset_validation.get_mask(continental_shelf = False)
    black_sea_mask_cs    = dataset_validation.get_mask(continental_shelf = True)
    black_sea_bathymetry = dataset_validation.get_depth(unit = "meter")

    # Used to detect the presence of hypoxia events
    hypoxia_treshold_standardized = dataset_train.get_treshold(standardized = True)

    # Creation of the dataloaders
    dataloader_train = BlackSea_Dataloader(dataset_train,
                                        window_input,
                                        window_output,
                                        frequencies,
                                        batch_size,
                                        num_workers,
                                        black_sea_mesh,
                                        black_sea_mask,
                                        black_sea_mask_cs,
                                        black_sea_bathymetry,
                                        random = True).get_dataloader()

    dataloader_valid = BlackSea_Dataloader_Diffusion(dataset_validation,
                                                    window_input,
                                                    window_output,
                                                    frequencies,
                                                    12,
                                                    num_workers,
                                                    black_sea_mesh,
                                                    black_sea_mask,
                                                    black_sea_mask_cs,
                                                    black_sea_bathymetry,
                                                    random = False).get_dataloader()
    # -------------—--------------------
    #     Neural Network & Training
    # -------------—--------------------
    # Initialization
    neural_net = Diffusion_UNET(window_input, window_output, diff_steps, diff_scheduler, diff_variance, scaling, frequencies, device).to(device)

    # Training Parameters
    optimizer  = optim.Adam(neural_net.parameters(), lr = learning_rate)
    scheduler  = LinearLR(optimizer, start_factor = 0.95, total_iters = nb_epochs)

    # Information about the model
    num_gpus  = torch.cuda.device_count()
    nn_params = neural_net.count_parameters()

    # Deploying the model on multiple GPUs
    neural_net.parrelize(num_gpus)

    # Displaying information over the terminal
    print("Total number of parameters: ", nn_params/1e6, "M")
    print("Available GPUs: ", num_gpus)

    # WandB (1) - Initialization of the run
    wandb.init(project = project, mode = mode, config = kwargs)

    # WandB (2) - Logging info
    wandb.config.update({"Number of Parameters": nn_params, "Number of GPUs": num_gpus})

    # ------- Training Loop -------
    for epoch in range(nb_epochs):

        # Stores the mean loss
        mean_loss = list()

        for conditioning, _, x in dataloader_train:

            # ------ Preprocessing -----
            #
            # Sampling uniformly diffusion steps
            diffusion_steps = torch.randint(0, diff_steps, (x.shape[0], 1))

            # Sampling noise
            noise = torch.normal(0, 1, x.shape)

            # Generating latent representations of the data
            z_t = neural_net.generate_latent(x, noise, diffusion_steps)

            # Pushing to device
            z_t, conditioning, noise, diffusion_steps =  z_t.to(device), conditioning.to(device),  noise.to(device), diffusion_steps

            # Adding the conditioning
            z_t = torch.cat([conditioning, z_t], dim = 1)

            # ----- Training -----
            #
            # Predicting the noise
            noise_pred = neural_net.predict(z_t, diffusion_steps)

            # Computing the loss (MSE between noise levels)
            loss = torch.pow(noise_pred[:, :, black_sea_mask_cs[0] == 1] - noise[:, :, black_sea_mask_cs[0] == 1], 2).nanmean()

            # Appending the loss
            mean_loss.append(loss.item())

            # WandB (4) - Logging the loss
            wandb.log({"Training/Loss (Instantaneous)": loss.item()})

            # Optimizing
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # WandB (3) - Logging the loss and the epoch
        wandb.log({"Training/Loss (Averaged Over Batch)": np.mean(mean_loss), "Epoch (Left)": nb_epochs - epoch})

        # Updating the learning rate
        scheduler.step()

        # Computing Metrics
        if epoch % 50 == 0:

            for c, _, x in dataloader_valid:

                # Pushing to device
                x, c = x.to(device), c.to(device)

                # Generating conditionnal samples
                forecast = neural_net.generate_samples(x = x, conditioning = c, number_trajectories = 24)

                # Computing Metrics
                metrics(x.cpu(), forecast.cpu(), black_sea_mask_cs, hypoxia_treshold_standardized)

                # Only on the first batch (= one year of data)
                break

    # Extracting the Neural Network back to CPU
    neural_net.to("cpu")

    # Finishing the Weight and Biases run
    wandb.finish()
