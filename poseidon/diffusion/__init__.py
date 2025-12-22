r"""Global constants for diffusion."""

# fmt: off
#
# Satellite Observation Model Parameters (mu_y, sigma_y)
#
#       y ~ N(A(x) + mu_y, sigma_y ** 2)
#
# Chlorophyll [mg/m³]
SAT_CHL_MU, SAT_CHL_STD = (
    [0.17, 0.0],
    [0.40, 0.0],
)

# Salinity [psu]
SAT_SAL_MU, SAT_SAL_STD = (
    [-0.07, 0.0],
    [ 0.91, 0.0],
)

# Temperature [°C]
SAT_TEMP_MU, SAT_TEMP_STD = (
    [0.09, 0.0],
    [0.51, 0.0],
)

# Sea Surface Height [m]
SAT_SSH_MU, SAT_SSH_STD = (
    [0.0,         0.0],
    [1.22 * 1e-3, 0.0],
)
