r"""Constants for diagnostics."""

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
    "CHL"      : "[mmol/m^3]",
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

CMAPS_SURF = {
    "DOX"      : cmo.haline,
    "CHL"      : cmo.curl_r,
    "vosaline" : cmo.balance,
    "votemper" : cmo.thermal,
    "ssh"      : cmo.diff,
}

CMAPS_LINE = {
    "DOX":"#225ea8",
    "CHL": "#238b45",
    "vosaline": "#8aa5bc",
    "votemper": "#e34a33",
    "ssh": "#b35806",
}

# A collection of ground truth dates for evaluating posterior sampling.
POSTERIOR_DATES = {
    0:  "2020-01-01",
    1:  "2020-02-01",
    2:  "2020-03-01",
    3:  "2020-04-01",
    4:  "2020-05-01",
    5:  "2020-06-01",
    6:  "2020-07-01",
    7:  "2020-08-01",
    8:  "2020-09-01",
    9:  "2020-10-01",
    10: "2020-11-01",
    11: "2020-12-01",
}
