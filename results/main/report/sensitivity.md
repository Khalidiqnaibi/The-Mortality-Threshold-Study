# Calibration sensitivity: coupling-offset spread σ_o

Levels s ∈ [0.5, 1.0, 1.5, 2.0, 3.0], 4+4 devices per cell (σ_o ≠ 0.01); baseline curve and s = 0 anchor from the main study. 4000 two-way cluster-bootstrap replicates.

|   sigma_o |   s_half |   s_half_lo |   s_half_hi |   area_um2 |     k |   R_inf |   R(s=0.5) |   R(s=1) |   R(s=2) |   min_own_acc |
|----------:|---------:|------------:|------------:|-----------:|------:|--------:|-----------:|---------:|---------:|--------------:|
|     0.003 |    1.543 |       1.391 |       1.678 |      0.420 | 4.915 |  -0.025 |      0.980 |    0.852 |    0.164 |         0.198 |
|     0.005 |    1.283 |       1.167 |       1.389 |      0.607 | 4.723 |   0.000 |      0.979 |    0.765 |    0.105 |         0.371 |
|     0.010 |    1.026 |       0.971 |       1.086 |      0.950 | 4.656 |  -0.032 |      0.969 |    0.530 |    0.016 |         0.124 |
|     0.020 |    0.698 |       0.652 |       0.744 |      2.055 | 4.328 |  -0.030 |      0.813 |    0.155 |   -0.025 |         0.335 |

log-log slope d ln s_half / d ln σ_o = -0.376 (0 = insensitive; −1 = s_half inversely proportional to σ_o)
