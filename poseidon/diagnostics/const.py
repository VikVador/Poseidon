r"""Constants for visualizations."""

import cmocean.cm as cmo

# fmt: off
#
TRANSLATION = {
    "DOX"      : "Oxygen",
    "CHL"      : "Chlorophyll",
    "vosaline" : "Salinity",
    "votemper" : "Temperature",
    "ssh"      : "Sea Surface Height",
}

TRANSLATION_LONG = {
    "DOX"      : "Oxygen",
    "CHL"      : "Chlorophyll",
    "vosaline" : "Sea Water Practical Salinity",
    "votemper" : "Sea Water Potential Temperature",
    "ssh"      : "Sea Surface Height Above Geoid",
}

UNITS = {
    "DOX"      : "[mmol/m^3]",
    "CHL"      : "[mg/m^3]",
    "vosaline" : "[PSU]",
    "votemper" : "[°C]",
    "ssh"      : "[m]",
}

CMAPS_LINE = {

    # Variables
    "DOX":"#225ea8",
    "CHL": "#238b45",
    "vosaline": "#8aa5bc",
    "votemper": "#e34a33",
    "ssh": "#b35806",

    # Distributions
    "p_x_d": "#4A5863",
    "p_x":"#828181",
}

CMAPS_SURF = {
    "DOX"      : cmo.haline,
    "CHL"      : cmo.curl_r,
    "vosaline" : cmo.balance,
    "votemper" : cmo.thermal,
    "ssh"      : cmo.diff,
}
