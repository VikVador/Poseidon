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
# A script to compute the distributions of a dataset.
#
#   Dawgz = False : compute the distributions over a given time period given by the user as arguments
#
#   Dawgz = True  : compute the distributions over all the possible time periods
#
import argparse

# Custom libraries
from dataset              import BlackSea_Dataset
from dataset_distribution import BlackSea_Dataset_Distribution

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

    # Date extension for result file
    date = f"8{start_year}-{start_month}_to_8{end_year}-{end_month}"

    # Information over terminal (1)
    print(f"Date: {date} - Loading the data")

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

    # Creation of a list containing all the datasets whose distribution must be analyze
    datasets = [data_oxygen_bottom, data_temperature, data_salinity, data_chlorophyll, data_klong, data_kshort]

    # ------- Extracting distributions from the data -------
    #
    # Information over terminal (2)
    print("Loading the distributions handler")

    # Loading distribution handler tool
    distribution_handler = BlackSea_Dataset_Distribution(subpopulation_percentage = 10,
                                                         dataloader = Dataset_physical,
                                                         datasets = datasets,
                                                         year_start = start_year,
                                                         year_end = end_year,
                                                         month_start = start_month,
                                                         month_end = end_month)

    # Information over terminal (3)
    print("Loading the marginals")

    # Computing marginal distributions
    distribution_handler.plot_marginal(save = True, file_name = f"../analysis/__distributions__/marginal/marginal_{date}.png")

    # Information over terminal (4)
    print("Loading the joints")

    # Computing joint distributions
    distribution_handler.plot_joint(save = True, file_name = f"../analysis/__distributions__/joint/joint_{date}.png")

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
    arguments = {
        'month_start': month_start[i],
        'month_end'  : month_end[i],
        'year_start' : year_start[i],
        'year_end'   : year_end[i]
    }

    # Launching the main
    main(arguments)

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
    USAGE:      python script_distribution.py --start_year    <X>
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
        schedule(compute_distribution, name = 'distributions', backend = 'slurm', export = 'ALL')

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