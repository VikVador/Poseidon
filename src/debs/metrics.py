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

# Pytorch
import torch

# Custom libraries
from dataset                import BlackSea_Dataset
from dataloader             import BlackSea_Dataloader


def analyze(y_true: torch.Tensor, y_pred: torch.Tensor, mask: np.array, dataset: BlackSea_Dataset, dataloader: BlackSea_Dataloader):
    """Used to compute a variety of metrics to evaluate the neural network"""

    def get_months_indices(data: np.array, window_output: int):
        """Used to extract the indices of the months, i.e. extract all the data for a given month"""

        # Conversion to torch
        data = torch.from_numpy(data)

        # Removing buffered days (beginning) and output window (end)
        data = data[365: -window_output]

        # Find the indices where the values change
        change_indices = torch.nonzero(data[:-1] != data[1:]).squeeze(1) + 1

        # Starting indices of consecutive sequences
        start_indices = torch.cat((torch.tensor([0]), change_indices))

        # Ending indices of consecutive sequences
        end_indices = torch.cat((change_indices, torch.tensor([len(data)])))

        return start_indices, end_indices

    def plot_seasonal_median_quantile(data: torch.Tensor, name_plot: str, name_wandb:str, ylimits: list, window_output: int, time_months: np.array, colors: list = ["#e13342", "#36193e", "#f6b48f"]):
        "Used to nicely plots the results"

        # Extracting the dimensions of the data
        samples, forecasted_days = data.shape

        # Extrating the indices of the months
        start_indices, end_indices = get_months_indices(time_months, window_output)

        # Axis coordinates definition
        row_axis, col_axis, row_dimension = list(), list(), [i for i in range(0, 12)]

        # Circular indices
        for i in range(4):
            row_axis += row_dimension
            col_axis += [i for j in range(0, 12)]

        # Looping over different limits
        for k, lims in enumerate(ylimits):

            # Initialization of the subplots (row = months, column = years)
            fig, axs = plt.subplots(12, 4, figsize=(18,26))

            for s, e, i, j in zip(start_indices, end_indices, row_axis, col_axis):

                # Extracting the data
                data_monthly = data[s:e]

                # Computing
                data_monthly_quantiled = torch.quantile(data_monthly, torch.tensor([0.10, 0.5, 0.90]), dim = 0)
                data_monthly_mean      = torch.mean(data_monthly, dim = 0)

                # Plotting the quantiles and median
                axs[i, j].plot(data_monthly_mean, "--",   color = colors[0], label='Mean')                                                # Mean
                axs[i, j].plot(data_monthly_quantiled[0], color = colors[2], label='Lower-quantile (10%)', marker = ".",  markersize = 4, ) # Lower-quantile
                axs[i, j].plot(data_monthly_quantiled[1], color = colors[1], label='Median (50%)')                                              # Median
                axs[i, j].plot(data_monthly_quantiled[2], color = colors[2], label='Upper-quantile (90%)', marker = "o",  markersize = 4, ) # Upper-quantile

                # Fixing limits
                axs[i, j].set_ylim(lims)

                # Filling the area between the quantiles
                axs[i, j].fill_between(range(10), data_monthly_quantiled[0], data_monthly_quantiled[2, :], color = colors[0], alpha = 0.1)

                # Adding a grid
                axs[i, j].grid(True, linestyle = '--', alpha = 0.25)

                # Remove x and y labels/ticks except for left edge and bottom edge
                if j not in [0]:
                    axs[i, j].set_yticklabels([])

                if j == 3:
                    axs[i, j].yaxis.tick_right()
                    axs[i, j].yaxis.set_label_position("right")

                if i not in [11]:
                    axs[i, j].set_xticklabels([])

                # Write the month corresponding to each row on the left edge
                if j == 3:
                    axs[i, j].set_ylabel(calendar.month_name[i+1], rotation = 270, labelpad = 15)

                # Add the xlabel
                if i in [11]:
                    axs[i, j].set_xlabel(f"Forecasted Days [{str(2016 + j)}]")

                # Adding metric name
                if i in [1, 4, 7, 10] and j == 0:
                    axs[i, j].set_ylabel(name_plot, labelpad = 20)

            # Create a legend subplot and add legend to it
            handles, labels = axs[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc = 'upper right', bbox_to_anchor=(0.905, 0.905), ncol = 4)

            # WandB - Sending Results
            wandb.log({f"Analyzing - Seasonal/{name_wandb}/Scale ({k})": wandb.Image(fig)})

    def plot_daily(y_true        : torch.Tensor,
                   y_pred        : torch.Tensor,
                   y_std         : torch.Tensor,
                   metric_BIAS   : torch.Tensor,
                   mask          : torch.Tensor,
                   window_output : int,
                   time_months   : np.array):
        """Used to show some daily predictions"""

        # Adding the mask
        y_true[:, :, mask == 0]      = float('nan')
        y_pred[:, :, mask == 0]      = float('nan')
        y_std[:, :, mask == 0]       = float('nan')
        metric_BIAS[:, :, mask == 0] = float('nan')

        # Extracting the index of each beginning month
        first_day_of_the_month, _ = get_months_indices(time_months, window_output)

        # Looping over the different years
        for year in range(4):

            # Initialization of the subplots (row = months, column = years)
            fig, axs = plt.subplots(12, 4, figsize = (14, 30))

            # Looping over the metrics then the day
            for i in range(12):

                # Determining the index of the first day of the month
                index = first_day_of_the_month[i * (year + 1)]

                # Plotting results
                axs[i, 0].imshow(y_true[     index,0], cmap = "viridis", vmin = -2,  vmax = 2)
                axs[i, 1].imshow(y_pred[     index,0], cmap = "viridis", vmin = -2,  vmax = 2)
                axs[i, 2].imshow(y_std[      index,0], cmap = "RdBu",    vmin = -2,  vmax = 2)
                axs[i, 3].imshow(metric_BIAS[index,0], cmap = "RdBu",    vmin = -15, vmax = 15)
                axs[i, 2].imshow(mask,                 cmap = "grey", alpha = 0.05)
                axs[i, 3].imshow(mask,                 cmap = "grey", alpha = 0.05)

                # Remove y ticks
                axs[i, 0].set_yticks([])
                axs[i, 1].set_yticks([])
                axs[i, 2].set_yticks([])
                axs[i, 3].set_yticks([])

                # Remove x ticks
                axs[i, 0].set_xticks([])
                axs[i, 1].set_xticks([])
                axs[i, 2].set_xticks([])
                axs[i, 3].set_xticks([])

                # Adding month of prediction
                axs[i, 3].yaxis.set_label_position("right")
                axs[i, 3].set_ylabel(calendar.month_name[i+1], rotation = 270, labelpad = 20)

            # Add colorbar every 3 rows
            for j, ax in enumerate(axs[-1]):
                fig.colorbar(ax.images[0], ax=ax, orientation='horizontal', fraction=0.065, pad=0.05)

            # Adding title
            axs[0, 0].set_title("Ground Truth",     pad = 10)
            axs[0, 1].set_title("Prediction",       pad = 10)
            axs[0, 2].set_title("Uncertainty",      pad = 10)
            axs[0, 3].set_title("Percent Bias [%]", pad = 10)

            # Wandb - Sending Everything
            wandb.log({f"Analyzing - Daily/{2016 + year}/Observation": wandb.Image(fig)})

    # --------------------
    #    Initialization
    # --------------------
    # Cropping area
    xmin, xmax, ymin, ymax = 20, 160, 55, 255

    # Extracting the mean and standard deviation
    y_pred_mean = y_pred[:, :, 0]
    y_pred_std  = torch.sqrt(torch.exp(y_pred[:, :, 1]))

    # Extracting the sea values
    y_pred_mean = y_pred_mean[:, :, mask[0] == 1]
    y_pred_std  = y_pred_std[:,  :, mask[0] == 1]

    y_true_spatial = y_true[:, :,xmin:xmax, ymin:ymax]
    y_true         = y_true[:, :, mask[0] == 1]
    y_true_mean    = torch.nanmean(y_true, axis = 2).unsqueeze(2)

    # ---------------------------------
    #        Metrics - Regression
    # ---------------------------------
    # Computing the metrics samplewise
    metric_MAE  = torch.nanmean(torch.absolute(y_true - y_pred_mean), axis = 2)
    metric_MSE  = torch.nanmean((y_true - y_pred_mean) ** 2, axis = 2)
    metric_RMSE = torch.sqrt(metric_MSE)
    metric_BIAS = torch.nanmean((y_true - y_pred_mean)/torch.absolute(y_true), axis = 2) * 100

    metric_R2_numerator   = torch.nansum((y_true - y_pred_mean) ** 2, axis = 2)
    metric_R2_denominator = torch.nansum((y_true - y_true_mean) ** 2, axis = 2)
    metric_R2             = 1 - metric_R2_numerator/metric_R2_denominator

    # 1. Global - Averaged value accross all samples and forecasted days
    metric_MAE_global  = torch.nanmean(metric_MAE)
    metric_MSE_global  = torch.nanmean(metric_MSE)
    metric_RMSE_global = torch.nanmean(metric_RMSE)
    metric_BIAS_global = torch.nanmean(metric_BIAS)
    metric_R2_global   = torch.nanmean(metric_R2)

    # 2. Global Forecasted Days - Averaged value accross all samples but for individual forecasted days
    metric_MAE_global_forecasted_days  = torch.nanmean(metric_MAE,  axis = 0)
    metric_MSE_global_forecasted_days  = torch.nanmean(metric_MSE,  axis = 0)
    metric_RMSE_global_forecasted_days = torch.nanmean(metric_RMSE, axis = 0)
    metric_BIAS_global_forecasted_days = torch.nanmean(metric_BIAS, axis = 0)
    metric_R2_global_forecasted_days   = torch.nanmean(metric_R2,   axis = 0)

    # Wandb - Sending global metrics
    wandb.log({
        "Analyzing - Global/Mean Absolute Error"              : metric_MAE_global.item(),
        "Analyzing - Global/Mean Squarred Error"              : metric_MSE_global.item(),
        "Analyzing - Global/Root Mean Squarred Error"         : metric_RMSE_global.item(),
        "Analyzing - Global/Percent Bias"                     : metric_BIAS_global.item(),
        "Analyzing - Global/Coefficient of Determination R^2" : metric_R2_global.item()
    })

    # Wandb - Sending global metrics for each forecasted days
    for i in range(metric_MAE_global_forecasted_days.shape[0]):
        wandb.log({
            f"Analyzing - Global/Forecasted Day {i}/Mean Absolute Error"              : metric_MAE_global_forecasted_days[i].item(),
            f"Analyzing - Global/Forecasted Day {i}/Mean Squarred Error"              : metric_MSE_global_forecasted_days[i].item(),
            f"Analyzing - Global/Forecasted Day {i}/Root Mean Squarred Error"         : metric_RMSE_global_forecasted_days[i].item(),
            f"Analyzing - Global/Forecasted Day {i}/Percent Bias"                     : metric_BIAS_global_forecasted_days[i].item(),
            f"Analyzing - Global/Forecasted Day {i}/Coefficient of Determination R^2" : metric_R2_global_forecasted_days[i].item()
        })

    # Extrating temporal information
    time_days, time_months, time_years = dataset.get_time()

    # 3. Seasonal - Averaged value accross all samples in a given month but for individual forecasted days
    #
    # Extrating temporal information
    time_days, time_months, time_years = dataset.get_time()

    # WandB - Plotting the results
    plot_seasonal_median_quantile(data        = metric_MAE,
                                name_plot     = "Mean Absolute Error (Avg. Days) [$mmol/m^3$]",
                                name_wandb    = "Mean Absolute Error",
                                ylimits       = [[0, 2], [0, 1], [0, 0.5]],
                                window_output = dataloader.window_output,
                                time_months   = time_months,
                                colors        = ["#98002e", "#36193e", "#180311"])

    plot_seasonal_median_quantile(data        = metric_MSE,
                                name_plot     = "Mean Squarred Error (Avg. Days) [$(mmol/m^3)^2$]",
                                name_wandb    = "Mean Squarred Error",
                                ylimits       = [[0, 2], [0, 1], [0, 0.5]],
                                window_output = dataloader.window_output,
                                time_months   = time_months,
                                colors        = ["#fdb917", "#36193e", "#180311"])

    plot_seasonal_median_quantile(data        = metric_RMSE,
                                name_plot     = "Root Mean Squarred Error (Avg. Days) [$mmol/m^3$]",
                                name_wandb    = "Root Mean Squarred Error",
                                ylimits       = [[0, 2], [0, 1], [0, 0.5]],
                                window_output = dataloader.window_output,
                                time_months   = time_months,
                                colors        = ["#132e52", "#36193e", "#180311"] )

    plot_seasonal_median_quantile(data        = metric_BIAS,
                                name_plot     = "Percent Bias [%]",
                                name_wandb    = "Percent Bias",
                                ylimits       = [[-500, 500], [-100, 100],  [-50, 50]],
                                window_output = dataloader.window_output,
                                time_months   = time_months,
                                colors        = ["#6c22a3", "#36193e", "#180311"] )

    plot_seasonal_median_quantile(data        = metric_R2,
                                name_plot     = "Coefficient of Determination $R^2$ (Avg. Days) [-]",
                                name_wandb    = "Coefficient of Determination $R^2$",
                                ylimits       = [[-5, 1.01], [-2, 1.01], [-0.5, 1.01]],
                                window_output = dataloader.window_output,
                                time_months   = time_months,
                                colors        = ["#2e5e4e", "#36193e", "#180311"] )

    # 4. Daily - Observing what is happening on a daily basis
    #
    # Extracting the data, keeping all the spatial information and cropping on the main interesting area
    y_pred_mean = y_pred[:, :, 0, xmin:xmax, ymin:ymax]
    y_pred_std  = torch.sqrt(torch.exp(y_pred[:, :, 1, xmin:xmax, ymin:ymax]))

    # Extracting the corresponding mask
    mask_daily = mask[0, xmin:xmax, ymin:ymax]

    # Computing the local bias
    metric_spatial_BIAS = (y_true_spatial - y_pred_mean)/torch.absolute(y_true_spatial) * 100

    # Plotting the results for the first day of each month for every year
    plot_daily(y_true        = y_true_spatial,
               y_pred        = y_pred_mean,
               y_std         = y_pred_std,
               metric_BIAS   = metric_spatial_BIAS,
               mask          = mask_daily,
               window_output = dataloader.window_output,
               time_months   = time_months)