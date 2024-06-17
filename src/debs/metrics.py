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
# A tool to compute a large variety of metrics on the outputs of a model.
#
import wandb
import calendar
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange, reduce, repeat

# Pytorch
import torch

# Custom libraries
from dataset                import BlackSea_Dataset
from dataloader             import BlackSea_Dataloader

class BlackSea_Metrics():
    r"""A tool to create a dataloader that processes and loads the Black Sea datasets on the fly"""

    def __init__(self, data_oxygen: np.array, mask: np.array, hypoxia_treshold: float, window_output: int = 10, number_trajectories: int = 10, number_samples: int = 1461):
        r"""Initialization of the metrics helper tool"""

        # Storing useful information
        self.index                        = 0
        self.mask                         = torch.from_numpy(mask)
        self.window_output                = window_output
        self.number_samples               = number_samples
        self.hypoxia_treshold             = hypoxia_treshold
        self.number_trajectories          = number_trajectories
        self.data_oxygen_temporal_average = torch.from_numpy(np.mean(data_oxygen[365:(1826 - window_output)], axis = 0))

        # Storing regression metrics results
        self.bias        = None
        self.mae         = None
        self.rmse        = None
        self.r2_spatial  = None
        self.r2_temporal = None

        # Storing classification metrics results
        self.acc  = None
        self.pre  = None
        self.rec  = None

    def analyze(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """Used to analyze the predictions of the neural network by computing different metrics"""

        def generate_trajectories(y_pred: torch.Tensor, number_trajectories: int):
            """Used to generate trajectories from the neural network means, standard deviations and coefficients"""

            # Extracting dimensions
            batch_size, forecasted_days, number_gaussians, values, x_res, y_res = y_pred.shape

            # ----- Deterministic Trajectories -----
            if number_gaussians == 1:
                return y_pred[:, :, :, 0].clone()

            # ----- Stochastic Trajectories -----
            #
            # Extracting values
            mean, std, pi = y_pred[:, :, :, 0], torch.exp(y_pred[:, :, :, 1]/2), torch.nn.functional.softmax(y_pred[:, :, :, 2], dim = 2)

            # Reshaping to apply multinomial
            mean = rearrange(mean, 'b d n x y -> (b d x y) n')
            std  = rearrange(std,  'b d n x y -> (b d x y) n')
            pi   = rearrange(pi,   'b d n x y -> (b d x y) n')

            # Sampling the coefficients
            coefficients = torch.multinomial(input = pi, num_samples = number_trajectories, replacement = True)

            # Extracting the mean and standard deviation of each trajectory
            mean = mean.gather(1, coefficients)
            std  = std.gather(1, coefficients)

            # Sampling the trajectories
            trajectories = torch.normal(mean, std)

            # Reshaping the trajectories
            return rearrange(trajectories, '(b d x y) n -> b d n x y', b = batch_size, d = forecasted_days, n = number_trajectories, x = x_res, y = y_res)

        def compute_percentiles(metric: torch.Tensor):
            """Used to compute the percentiles (10% and 90%) as well as the median across realizations for given metric results"""
            return rearrange(torch.quantile(metric, torch.tensor([0.10, 0.5, 0.90]), dim = 2), 'n b d -> b d n')

        def vizualize_trajectories(trajectories: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor, index: int):
            """Used to vizualize different generated the trajectories"""

            # -------------------------
            #    Deterministic Model
            # -------------------------
            #
            # Only one trajectory because its the mean
            if trajectories.shape[2] == 1:

                # Extracting trajectories
                visualized_trajectory = trajectories[0, 0, 0].detach().numpy()
                visualized_true         = y_true[0, 0].detach().numpy()

                # Masking the values
                visualized_trajectory[mask[0] == 0] = np.nan
                visualized_true[mask[0] == 0]       = np.nan

                # Extracting region of interest
                visualized_trajectory = visualized_trajectory[25:125, 70:270]
                visualized_true       = visualized_true[25:125, 70:270]

                # Defining minimum and maximum values
                vmin, vmax = np.nanmin(visualized_true), np.nanmax(visualized_true)

                # Creation of the Plot
                fig, axes = plt.subplots(1, 2, figsize = (15, 6))
                im = axes[0].imshow(visualized_true, cmap = "viridis", vmin=vmin, vmax=vmax)
                axes[0].set_title("Ground Truth", fontsize = 10)
                axes[0].axis('off')
                im = axes[1].imshow(visualized_trajectory, cmap = "viridis", vmin=vmin, vmax=vmax)
                axes[1].set_title("Prediction (Mean)", fontsize = 10)
                axes[1].axis('off')

                # Add a colorbar to the ground truth plot
                cbar = fig.colorbar(im, ax=axes[1], fraction = 0.026, pad = 0.04)

                # Sending results to WandB
                wandb.log({f"Observing Trajectories (Mean)/Sample ({index})": wandb.Image(fig)})
                plt.close()

            # -------------------------
            #     Generative Model
            # -------------------------
            #
            else:

                # Extracting trajectories
                visualized_trajectories = trajectories[0, 0, :10].detach().numpy()
                visualized_true         = y_true[0, 0].detach().numpy()

                # Masking the values
                visualized_trajectories[:, mask[0] == 0] = np.nan
                visualized_true[mask[0] == 0]         = np.nan

                # Extracting region of interest
                visualized_trajectories = visualized_trajectories[:, 25:125, 70:270]
                visualized_true         = visualized_true[25:125, 70:270]

                # Defining minimum and maximum values
                vmin, vmax = np.nanmin(visualized_true), np.nanmax(visualized_true)

                # Create a figure with 3 rows and 5 columns
                fig, axes = plt.subplots(3, 4, figsize = (15, 6))

                # Plot the ground truth in the top-left corner
                im = axes[0, 0].imshow(visualized_true, cmap = "viridis", vmin=vmin, vmax=vmax)
                axes[0, 0].set_title("Ground Truth", fontsize = 10)
                axes[0, 0].axis('off')

                # Add a colorbar to the ground truth plot
                cbar = fig.colorbar(im, ax=axes[0, 0], fraction = 0.026, pad = 0.04)

                # Hide the rest of the plots in the first row
                for i in range(1, 4):
                    axes[0, i].axis('off')

                # Plot the predicted trajectories in the next two rows
                for i in range(8):
                    row = i // 4 + 1
                    col = i % 4
                    axes[row, col].imshow(visualized_trajectories[i], cmap = "viridis", vmin=vmin, vmax=vmax)
                    axes[row, col].axis('off')

                # Sending results to WandB
                wandb.log({f"Observing Trajectories/Sample ({index})": wandb.Image(fig)})
                plt.close()

        # Generating the trajectories
        trajectories = generate_trajectories(y_pred, self.number_trajectories)

        # Visualizing the trajectories on WandB
        vizualize_trajectories(trajectories, y_true, self.mask, self.index)

        # Updating the index (used to name the different samples)
        self.index += 1

        # Masking useless values
        trajectories = trajectories[:, :, :, self.mask[0] == 1]
        y_true       = y_true[:, :, self.mask[0] == 1]

        # Adding dimensions to make broadcasting possible
        y_true = y_true[:, :, None, :].expand(-1, -1, trajectories.shape[2], -1)

        # computing the oxygen temporal and spatial average
        data_oxygen_temporal_average = self.data_oxygen_temporal_average[None, None, None, self.mask[0] == 1].expand(trajectories.shape)
        data_oxygen_spatial_average  = torch.nanmean(y_true, axis = 3, keepdim = True).expand(trajectories.shape)

        # -------------------------
        #    Metrics Regression
        # -------------------------
        #
        # Computing the different metrics
        metric_MAE  = torch.nanmean(torch.absolute(y_true - trajectories),          axis = 3)
        metric_BIAS = torch.nanmean((y_true - trajectories)/torch.absolute(y_true), axis = 3) * 100
        metric_RMSE = torch.sqrt(torch.nanmean((y_true - trajectories) ** 2,        axis = 3))

        # Computing the R2 scores spatially and temporally
        metric_R2_NUMERATOR = torch.sum((y_true - trajectories) ** 2, axis = 3)
        metric_R2_TEMPORAL  = 1 - metric_R2_NUMERATOR / torch.sum((y_true - data_oxygen_temporal_average) ** 2, axis = 3)
        metric_R2_SPATIAL   = 1 - metric_R2_NUMERATOR / torch.sum((y_true - data_oxygen_spatial_average)  ** 2, axis = 3)

        # Computing the quantiles and updating the metrics
        self.mae         = compute_percentiles(metric_MAE)         if self.mae is None         else torch.cat((self.mae,         compute_percentiles(metric_MAE)),         dim = 0)
        self.bias        = compute_percentiles(metric_BIAS)        if self.bias is None        else torch.cat((self.bias,        compute_percentiles(metric_BIAS)),        dim = 0)
        self.rmse        = compute_percentiles(metric_RMSE)        if self.rmse is None        else torch.cat((self.rmse,        compute_percentiles(metric_RMSE)),        dim = 0)
        self.r2_spatial  = compute_percentiles(metric_R2_SPATIAL)  if self.r2_spatial is None  else torch.cat((self.r2_spatial,  compute_percentiles(metric_R2_SPATIAL)),  dim = 0)
        self.r2_temporal = compute_percentiles(metric_R2_TEMPORAL) if self.r2_temporal is None else torch.cat((self.r2_temporal, compute_percentiles(metric_R2_TEMPORAL)), dim = 0)

        # ----------------------------
        #    Metrics Classification
        # ----------------------------
        #
        # Detecting Hypoxia
        trajectories = (trajectories < self.hypoxia_treshold) * 1
        y_true       = (y_true       < self.hypoxia_treshold) * 1

        # Calculate True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN)
        TP = torch.sum(y_true * trajectories,             dim = 3)
        FP = torch.sum((1 - y_true) * trajectories,       dim = 3)
        TN = torch.sum((1 - y_true) * (1 - trajectories), dim = 3)
        FN = torch.sum(y_true * (1 - trajectories),       dim = 3)

        # Computing the metrics
        metric_ACC = (TP + TN) / (TP + TN + FP + FN)
        metric_PRE = TP / (TP + FP)
        metric_REC = TP / (TP + FN)

        # Computing the quantiles and updating the metrics
        self.acc = compute_percentiles(metric_ACC) if self.acc is None else torch.cat((self.acc, compute_percentiles(metric_ACC)), dim = 0)
        self.pre = compute_percentiles(metric_PRE) if self.pre is None else torch.cat((self.pre, compute_percentiles(metric_PRE)), dim = 0)
        self.rec = compute_percentiles(metric_REC) if self.rec is None else torch.cat((self.rec, compute_percentiles(metric_REC)), dim = 0)

    def send_results(self):

        def metric_global(data: torch.Tensor, name: str):
            """Used to compute a global metric, i.e., average over samples and then forecast"""

            # Computing the global metric
            global_metric = torch.nanmean(data, axis = (0, 1))

            # Storing results for WandB
            return {f"Global Metrics/{name} (10%)":  global_metric[0],
                    f"Global Metrics/{name} (50%)":  global_metric[1],
                    f"Global Metrics/{name} (90%)":  global_metric[2]}

        def metric_forecast(data: torch.Tensor, name: str):
            """Used to compute a forecast metric, i.e., average over samples"""

            # Computing the forecast metric
            forecast_metric = torch.nanmean(data, axis = (0))

            # Stores the complete dictionary of results
            forecast_results = {}

            # Computing results for each day
            for f, results in enumerate(forecast_metric):
                forecast_results[f"Forecast Metrics/{name} - Day " + str(f) + " (10%)"] = results[0]
                forecast_results[f"Forecast Metrics/{name} - Day " + str(f) + " (50%)"] = results[1]
                forecast_results[f"Forecast Metrics/{name} - Day " + str(f) + " (90%)"] = results[2]

            return forecast_results

        def metric_forecast_evolution(data: torch.Tensor, limits: list, name: str):
            """Used to display the evolution of a metric over time for a single, i.e. the validation"""

            # Conversion to numpy
            data = data.detach().numpy()

            # Retrieving dimensions for ease of use
            number_days, forecasted_days, values = data.shape

            # -------------------------------
            #   Plots For Individual Model
            # -------------------------------
            #
            # Due to WandB restriction, by sending a plot, we can display evolution
            # over epochs but cannot plot model results against one another
            #
            # Looping over different limits
            for i, l in enumerate(limits):

                # Plotting the evolution of the metric
                fig = plt.figure(figsize = (15, 5))

                # Showing the best (first day), worst (last day) forecast
                plt.plot(data[:,  0,  0],  color = "#00ffff", linestyle = "dotted", label = f'($T_{0}$) Q10%')
                plt.plot(data[:,  0,  1],  color = "#00ffff", linestyle = "solid",  label = f'($T_{0}$) Median')
                plt.plot(data[:,  0,  2],  color = "#00ffff", linestyle = "dashed", label = f'($T_{0}$) Q90%')
                plt.plot(data[:,  -1,  0], color = "#004c6d", linestyle = "dotted", label = f'($T_{forecasted_days - 1}$) Q10%')
                plt.plot(data[:,  -1,  1], color = "#004c6d", linestyle = "solid",  label = f'($T_{forecasted_days - 1}$) Median')
                plt.plot(data[:,  -1,  2], color = "#004c6d", linestyle = "dashed", label = f'($T_{forecasted_days - 1}$) Q90%')
                plt.grid(alpha = 0.5)
                plt.xlabel("Days")
                plt.ylabel(name)
                plt.ylim(l)
                plt.legend(loc = 'upper right', bbox_to_anchor=(1.01, 1.15), ncol = 6)

                # Sending to WandB
                wandb.log({f"Forecast Metrics Evolution/{name} ({i})": wandb.Image(fig)})
                plt.close()

            # -------------------------------
            #   Comparison Plot For Models
            # -------------------------------
            #
            # Days on the x-axis
            x_axis = np.arange(number_days)

            # Logging the results
            wandb.log({f"Forecast Metrics Evolution (Comparison)/{name} ($T_{0}$)" : wandb.plot.line_series(
                                        xs = x_axis,
                                        ys = [data[:, 0, 0], data[:, 0, 1], data[:, 0, 2]],
                                    keys = ["Q10%", "Median", "Q90%"],
                                    title = f"{name} - T0",
                                    xname = "Days"),
                        f"Forecast Metrics Evolution (Comparison)/{name} ($T_{forecasted_days - 1}$)" : wandb.plot.line_series(
                                        xs = x_axis,
                                        ys = [data[:, -1, 0], data[:, -1, 1], data[:, -1, 2]],
                                    keys = ["Q10%", "Median", "Q90%"],
                                    title = f"{name} - T{forecasted_days - 1}",
                                    xname = "Days")})

        # Global - Give a rough idea of the performance
        wandb.log(metric_global(self.mae,         "Mean Absolute Error"))
        wandb.log(metric_global(self.bias,        "Percent Bias"))
        wandb.log(metric_global(self.rmse,        "Root Mean Square Error"))
        wandb.log(metric_global(self.r2_spatial,  "Coefficient of Determination R2 - Spatial"))
        wandb.log(metric_global(self.r2_temporal, "Coefficient of Determination R2 - Temporal"))
        wandb.log(metric_global(self.acc,         "Accuracy"))
        wandb.log(metric_global(self.pre,         "Precision"))
        wandb.log(metric_global(self.rec,         "Recall"))

        # Forecast - Give a rough idea of the performance for each forecasted days
        wandb.log(metric_forecast(self.mae,         "Mean Absolute Error"))
        wandb.log(metric_forecast(self.bias,        "Percent Bias"))
        wandb.log(metric_forecast(self.rmse,        "Root Mean Square Error"))
        wandb.log(metric_forecast(self.r2_spatial,  "Coefficient of Determination R2 - Spatial"))
        wandb.log(metric_forecast(self.r2_temporal, "Coefficient of Determination R2 - Temporal"))
        wandb.log(metric_forecast(self.acc,         "Accuracy"))
        wandb.log(metric_forecast(self.pre,         "Precision"))
        wandb.log(metric_forecast(self.rec,         "Recall"))

        # Forecast Evolution - Give an idea of the evolution of a metric accross the validation set for the best and worst forecast
        metric_forecast_evolution(self.mae,         [[0, 1], [0, 2], [0, 4]],                "Mean Absolute Error")
        metric_forecast_evolution(self.bias,        [[-100, 100], [-250, 100], [-500, 100]], "Percent Bias")
        metric_forecast_evolution(self.rmse,        [[0, 1], [0, 2], [0, 4]],                "Root Mean Square Error")
        metric_forecast_evolution(self.r2_spatial,  [[-25, 1.01], [-5, 1.01], [-2, 1.01]],   "Coefficient of Determination R2 - Spatial")
        metric_forecast_evolution(self.r2_temporal, [[-25, 1.01], [-5, 1.01], [2, 1.01]],    "Coefficient of Determination R2 - Temporal")
        metric_forecast_evolution(self.acc,         [[0, 0.25], [0, 0.5], [0, 1.01]],        "Accuracy")
        metric_forecast_evolution(self.pre,         [[0, 0.25], [0, 0.5], [0, 1.01]],        "Precision")
        metric_forecast_evolution(self.rec,         [[0, 0.25], [0, 0.5], [0, 1.01]],        "Recall")
