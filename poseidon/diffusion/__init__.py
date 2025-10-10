r"""Global constants for diffusion."""

# fmt: off
#
# Satellite Observation Model Parameters (mu_y, sigma_y)
#
#       y ~ N(A(x) + mu_y, sigma_y ** 2)
#
SAT_CHL_MU, SAT_CHL_STD = (
    [0.1699, 0.0],
    [0.4049, 0.0],
)

SAT_SAL_MU, SAT_SAL_STD = (
    [0.0, 0.0],
    [0.5, 0.1],
)

SAT_TEMP_MU, SAT_TEMP_STD = (
    [-0.09, 0.13],
    [0.69,   0.1],
)

SAT_SSH_MU, SAT_SSH_STD = (
    [0.0,         0.0],
    [1.22 * 1e-3, 0.0],
)
