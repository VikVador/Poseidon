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
# A script to extract the path of all files and save it as a .txt files. Since there
# is such an heterogeneity in the ways data files are saved, it is important to make
# sure everything is loaded perfectly.
#
import os
import json


def get_files_name(folder: str, keyword: str):
    """Get all the files name in a folder that contain a specific keyword"""
    return [f for f in os.listdir(folder) if keyword in f]

def process_dictionary(smart_dict: dict):
    """Process a dictionary and keep only the first value for keys with two values"""

    # Looping each combinations
    for key, values in smart_dict.items():

        # If there is a duplicate, removes it !
        if len(values) == 2:
            smart_dict[key] = [values[0]]

    return smart_dict

def create_dictionary(smart_dict: dict, files: list, folder : str, separator: str = "_grid_T_"):
    """Create a dictionary with the month and day of each file as keys and the name of the files as values"""

    # Loop through the years
    for year in range(1980, 2023):

        # Loop through the months
        for month in range(1, 13):

            # Generate indicator for the year-month
            indicator = f"{year}{month:02}"

            # Generate key for the dictionary
            key = f"{year}-{month:02}"

            # Temporary storage for file names
            file_names = []

            # Loop through the files
            for file_name in files:

                # Extract file name after the separator
                file_processed = file_name.split(separator)[1].split("_")[0][:6]

                # Check if the indicator is in the file name
                if indicator in file_processed and file_name.endswith(".nc4"):

                    # Add the file to the list
                    file_names.append(folder + file_name)

            # Add the month to the dictionary if files are found
            if file_names:
                smart_dict[key] = file_names

    return smart_dict

# ------------------------
#       MAIN SCRIPT
# ------------------------
if __name__ == "__main__":

    # Contains all the years of data possible (in the output_HR001 folder)
    years_list = ['1980-1990', '1991-1997', '1997', '1997-part1', '1998', '1999', '2000',
                  '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009',
                  '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018',
                  '2019', '2020', '2021', '2022']

    # Path to this folder (if this code is executed in src folder)
    main_path  = "../../../../../../../projects/acad/bsmfc/nemo4.2.0/BSFS_BIO/output_HR001/"

    # Stores all the information as a nice dictionnary, one for the physics and the other for biogeochemistry !
    paths_phy, paths_bio = dict(), dict()

    # Looping over the different folders
    for y in years_list:

        # Complete path to the simulation folder
        full_path = f"{main_path}{y}/"

        # Retrieving all the files name
        names_phy = sorted(get_files_name(full_path, "grid_T"))
        names_bio = sorted(get_files_name(full_path, "ptrc_T"))

        # Creation of the dictionary
        paths_phy = create_dictionary(smart_dict = paths_phy,
                                           files = names_phy,
                                          folder = f"/{y}/",
                                       separator = "_grid_T_")

        paths_bio = create_dictionary(smart_dict = paths_bio,
                                           files = names_bio,
                                          folder = f"/{y}/",
                                      separator = "_ptrc_T_")

    # Removing duplicates
    paths_phy = process_dictionary(paths_phy)
    paths_bio = process_dictionary(paths_bio)

    # Saving them as helper files for loading data
    with open('../../information/grid_T.txt', 'w') as file:
        json.dump(paths_phy, file)

    with open('../../information/ptrc_T.txt', 'w') as file:
        json.dump(paths_bio, file)