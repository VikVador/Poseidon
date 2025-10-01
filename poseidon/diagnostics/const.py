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
    "vosaline" : "[1e-3]",
    "votemper" : "[degC]",
    "ssh"      : "[m]",
}

INTERVALS = {
    "DOX"      : (-50, 400),
    "CHL"      : (-1,  3),
    "vosaline" : (16, 23),
    "votemper" : (1, 30),
    "ssh"      : (-0.5, 0.5),
}


CMAPS_LINE = {
    "DOX":"#225ea8",
    "CHL": "#238b45",
    "vosaline": "#8aa5bc",
    "votemper": "#e34a33",
    "ssh": "#b35806",
}

CMAPS_SURF = {
    "DOX"      : cmo.haline,
    "CHL"      : cmo.curl_r,
    "vosaline" : cmo.balance,
    "votemper" : cmo.thermal,
    "ssh"      : cmo.diff,
}
