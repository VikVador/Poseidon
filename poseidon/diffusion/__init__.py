r"""Global constants for diffusion."""

# fmt: off
#
# Satellite Observational Error (mean, std)
SAT_CHL_BIAS, SAT_CHL_STD = (
    [0.1699, 0.0],
    [0.4049, 0.0],
)

SAT_SAL_BIAS, SAT_SAL_STD = (
    [0.0, 0.0],
    [0.5, 0.1],
)

SAT_TEMP_BIAS, SAT_TEMP_STD = (
    [-0.09, 0.13],
    [0.69,   0.1],
)

SAT_SSH_BIAS, SAT_SSH_STD = (
    [0.0,         0.0],
    [1.22 * 1e-3, 0.0],
)
