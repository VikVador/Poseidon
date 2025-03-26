r"""Constants used in the diagnostics module for plotting."""

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

CMAPS_SURF = {
    "DOX"      : "RdYlBu_r",
    "CHL"      : cmo.curl_r,
    "vosaline" : cmo.balance,
    "votemper" : cmo.thermal,
    "ssh"      : cmo.diff,
}
