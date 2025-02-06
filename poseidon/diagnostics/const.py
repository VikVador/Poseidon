r"""Constants used in the diagnostics module for plotting."""

import cmocean.cm as cmo

# fmt: off
#
# Translation of the variables
TRANSLATION = {
    "DOX"      : "Oxygen",
    "CHL"      : "Chlorophyll",
    "vosaline" : "Salinity",
    "votemper" : "Temperature",
    "ssh"      : "Sea Surface Height",
}

# Colormaps used for surface plots
CMAPS_SURF = {
    "DOX"      : "RdYlBu_r",
    "CHL"      : cmo.curl_r,
    "vosaline" : cmo.balance,
    "votemper" : cmo.thermal,
    "ssh"      : cmo.diff,
}
