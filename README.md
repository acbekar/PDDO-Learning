# PDDO-Learning
PDDO-Learning: Peridynamics Enabled Learning Partial Differential Equations

This study presents an approach to discover the significant terms in partial differential equations (PDEs) that describe particular phenomena based on the measured data. The relationship between the known field data and its continuous representation of PDEs is achieved through a linear regression model. It specifically employs the peridynamic differential operator (PDDO) and sparse linear regression learning algorithm. The PDEs are approximated by constructing a feature matrix, velocity vector and unknown coefficient vector. Each candidate term (derivatives) appearing in the feature matrix is evaluated numerically by using the PDDO. The solution to the regression model with regularization is achieved through Douglas-Rachford (D-R) algorithm which is based on proximal operators. This coupling performs well due to their robustness to noisy data and the calculation of accurate derivatives. Its effectiveness is demonstrated by considering several fabricated data associated with challenging nonlinear PDEs such as Burgers, Swift-Hohenberg (S-H), Korteweg-de Vries (KdV), Kuramoto-Sivashinsky (K-S), nonlinear Schrödinger (NLS) and Cahn-Hilliard (C-H) equations.

## Citation
If you find our work useful in your research, please cite:
```
@article{BEKAR2021110193,
title = {Peridynamics enabled learning partial differential equations},
journal = {Journal of Computational Physics},
volume = {434},
pages = {110193},
year = {2021},
issn = {0021-9991},
doi = {https://doi.org/10.1016/j.jcp.2021.110193},
url = {https://www.sciencedirect.com/science/article/pii/S0021999121000887},
author = {Ali C. Bekar and Erdogan Madenci},
keywords = {Partial differential equations, Machine learning, Peridynamics, Sparse optimization}
}
```

## Contact
If you have any questions, please feel free to email <acbekar@email.arizona.edu>.
