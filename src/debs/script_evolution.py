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
# A script to compute the evolution of variables in a dataset over a given time period.
#
#   Dawgz = False : compute the evolution over a given time period given by the user as arguments
#
#   Dawgz = True  : compute the evolution over all the possible time periods
#
import argparse

# Custom libraries
from dataset              import BlackSea_Dataset
from dataset_evolution    import BlackSea_Dataset_Evolution

# Dawgz library (used to parallelized the jobs)
from dawgz import job, schedule

# ---------------------------------------------------------------------
#
#                              MAIN FUNCTION
#
# ---------------------------------------------------------------------
def main(**kwargs):

    # ------- Arguments -------
    start_month = kwargs['month_start']
    end_month   = kwargs['month_end']
    start_year  = kwargs['year_start']
    end_year    = kwargs['year_end']

    # Information over terminal (1)
    print("Loading the data")

    # Intialization of the dataset handler !
    Dataset_physical = BlackSea_Dataset(year_start = start_year, year_end = end_year, month_start = start_month,  month_end = end_month, variable = "grid_T")
    Dataset_bio      = BlackSea_Dataset(year_start = start_year, year_end = end_year, month_start = start_month,  month_end = end_month, variable = "ptrc_T")

    # Loading the different field values
    data_temperature   = Dataset_physical.get_temperature()
    data_salinity      = Dataset_physical.get_salinity()
    data_oxygen_bottom = Dataset_bio.get_oxygen_bottom()
    data_chlorophyll   = Dataset_bio.get_chlorophyll()
    data_kshort        = Dataset_bio.get_light_attenuation_coefficient_short_waves()
    data_klong         = Dataset_bio.get_light_attenuation_coefficient_long_waves()

    # ------- Extracting information from the data -------
    #
    # Information over terminal (2)
    print("Loading the tools")

    # Loading the dataset tool !
    tool_temperature = BlackSea_Dataset_Evolution(Dataset_physical, data_temperature)
    tool_salinity    = BlackSea_Dataset_Evolution(Dataset_physical, data_salinity)
    tool_oxygen      = BlackSea_Dataset_Evolution(Dataset_bio,      data_oxygen_bottom)
    tool_chlorophyll = BlackSea_Dataset_Evolution(Dataset_bio,      data_chlorophyll)
    tool_kshort      = BlackSea_Dataset_Evolution(Dataset_bio,      data_kshort)
    tool_klong       = BlackSea_Dataset_Evolution(Dataset_bio,      data_klong)

    # Date extension for result file
    date = f"{start_month}-8{start_year}_to_{end_month}-8{end_year}"

    # ------- Evolution plots -------
    #
    # Information over terminal (3)
    print("Evolution plots")

    tool_temperature.plot_line("Temperature [C°]" ,     save = True, file_name = f"../../analysis/temperature/evolution/temperature_{date}.png")
    tool_salinity.plot_line(      "Salinity [ppt]",     save = True, file_name = f"../../analysis/salinity/evolution/salinity_{date}.png")
    tool_oxygen.plot_line(          "Oxygen [mmol/m3]", save = True, file_name = f"../../analysis/oxygen/evolution/oxygen_{date}.png")
    tool_chlorophyll.plot_line("Chlorophyll [mmol/m3]", save = True, file_name = f"../../analysis/chlorophyll/evolution/chlorophyll_{date}.png")
    tool_klong.plot_line(           "K-Long [-]",       save = True, file_name = f"../../analysis/klong/evolution/klong_{date}.png")
    tool_kshort.plot_line(         "K-Short [-]",       save = True, file_name = f"../../analysis/kshort/evolution/kshort_{date}.png")

    # ------- Mask plots -------
    #
    # Information over terminal (4)
    print("Mask plot")

    tool_oxygen.plot_treshold(save = True, file_name = f"../analysis/oxygen/treshold/oxygen_{date}.png")

    # ------- Animation plots -------
    #
    # Information over terminal 5()
    print("Animation")

    tool_temperature.plot_animation(f"../../analysis/temperature/animation/temperature_{date}.gif", ylabel = "Temperature [C°]")
    tool_salinity.plot_animation(      f"../../analysis/salinity/animation/salinity_{date}.gif",    ylabel = "Salinity [ppt]")
    tool_oxygen.plot_animation(          f"../../analysis/oxygen/animation/oxygen_{date}.gif",      ylabel = "Oxygen [mmol/m3]")
    tool_chlorophyll.plot_animation(f"../../analysis/chlorophyll/animation/chlorophyll_{date}.gif", ylabel = "Chlorophyll [mmol/m3]")
    tool_klong.plot_animation(            f"../../analysis/klong/animation/klong_{date}.gif",       ylabel = "K-Long [-]")
    tool_kshort.plot_animation(          f"../../analysis/klong/animation/kshort_{date}.gif",       ylabel = "K-Short [-]")

# ---------------------------------------------------------------------
#
#                                  DAWGZ
#
# ---------------------------------------------------------------------
#
# -------------
# Possibilities
# -------------
# 1) Distributions over the 10 years
month_start = [1]
month_end   = [12]
year_start  = [1]
year_end    = [9]

# 2) Distributions over the years
for i in range(1, 9):
    month_start.append(1)
    month_end.append(12)
    year_start.append(i)
    year_end.append(i + 1)

# 3) Distribution over the months
for i in range(1, 9):
    for j in range(1, 12):
        month_start.append(j)
        month_end.append(j + 1)
        year_start.append(i)
        year_end.append(i)

# Total number of jobs
nb_tasks = len(month_start)

# ----
# Jobs
# ----
@job(array = nb_tasks, cpus = 1, ram = '64GB', time = '12:00:00', project = 'bsmfc', user = 'vmangeleer@uliege.be', type = 'FAIL')
def compute_distribution(i: int):

    # Retreiving corresponding time period
    dates = {
        'month_start': month_start[i],
        'month_end'  : month_end[i],
        'year_start' : year_start[i],
        'year_end'   : year_end[i]
    }

    # Launching the main
    main(dates)

# ---------------------------------------------------------------------
#
#                                  MAIN
#
# ---------------------------------------------------------------------
if __name__ == "__main__":

    # ------- Parsing the command-line arguments -------
    #
    # Definition of the help message that will be shown on the terminal
    usage = """
    USAGE:      python script_evolution.py --start_year    <X>
                                           --end_year      <X>
                                           --start_month   <X>
                                           --end_month     <X>
                                           --dawgz         <X>
    """
    # Initialization of the parser
    parser = argparse.ArgumentParser(usage)

    # Definition of the possible stuff to be parsed
    parser.add_argument(
        '--start_year',
        help    = 'Starting year to collect data',
        type    = int,
        default = 0)

    parser.add_argument(
        '--end_year',
        help    = 'Ending year to collect data',
        type    = int,
        default = 0)

    parser.add_argument(
        '--start_month',
        help    = 'Starting month to collect data',
        type    = int,
        default = 1)

    parser.add_argument(
        '--end_month',
        help    = 'Ending month to collect data',
        type    = int,
        default = 2)

    parser.add_argument(
        '--dawgz',
        help    = 'Determine if the script is run with dawgz or not',
        type    = str,
        default = "False",
        choices = ['True', 'False'])

    # Retrieving the values given by the user
    args = parser.parse_args()

    # ------- Running with dawgz -------
    if args.dawgz == "True":

        # Information over terminal
        print("Running with dawgz")

        # Running the jobs
        schedule(compute_distribution, name='distributions', backend='slurm', export='ALL')

    # ------- Running without dawgz -------
    else:

        # Information over terminal
        print("Running without dawgz")

        # Retreiving corresponding time period
        arguments = {
            'month_start': args.start_month,
            'month_end'  : args.end_month,
            'year_start' : args.start_year,
            'year_end'   : args.end_year
        }

        # Launching the main
        main(**arguments)

        # Information over terminal
        print("Done")