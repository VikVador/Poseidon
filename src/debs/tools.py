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
# Contains different random tools used in the project (e.g. generating fake datasets for testing, loading local or cluster path, ...)
#


def project_title(kwargs : dict):
    r"""Used to display information about the run"""
    print("-------------------------------------------------------")
    print("                                                       ")
    print("                    ESA - PROJECT                      ")
    print("                                                       ")
    print("          BLACK SEA DEOXYGENATION EMULATOR             ")
    print("                                                       ")
    print("-------------------------------------------------------")
    print("                                                       ")
    for k, v in kwargs.items():
        print(f"- {k} : {v}")
    print("\n-----------------")
    print("Emulator Training")
    print("-----------------")
