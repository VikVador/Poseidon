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
    "DOX":      "#0077B6",
    "CHL":      "#40C88C",
    "vosaline": "#BF3F64",
    "votemper": "#E67E22",
    "ssh":      "#9C7AFF",
    "p_x_d":    "#4A5863",
    "p_x":      "#828181",
}

CMAPS_SURF = {
    "DOX"      : cmo.haline,
    "CHL"      : cmo.curl_r,
    "vosaline" : cmo.balance,
    "votemper" : cmo.thermal,
    "ssh"      : cmo.diff,
}
