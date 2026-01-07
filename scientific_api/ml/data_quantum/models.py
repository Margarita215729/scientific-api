"""
Quantum system models for graph generation.

This module implements parametrized quantum systems:
- Two-dimensional harmonic oscillator
- Two-dimensional potential well with local perturbations
"""

from typing import Tuple

import numpy as np
from scipy import sparse

from app.core.logging import get_logger

logger = get_logger(__name__)


def harmonic_oscillator_2d_potential(
    x: np.ndarray,
    y: np.ndarray,
    omega_x: float = 1.0,
    omega_y: float = 1.0,
    x0: float = 0.0,
    y0: float = 0.0,
) -> np.ndarray:
    """
    Compute 2D harmonic oscillator potential on a grid.

    V(x, y) = 0.5 * omega_x^2 * (x - x0)^2 + 0.5 * omega_y^2 * (y - y0)^2

    Args:
        x: X-coordinate grid (2D array).
        y: Y-coordinate grid (2D array).
        omega_x: Angular frequency in x direction.
        omega_y: Angular frequency in y direction.
        x0: Center position in x direction.
        y0: Center position in y direction.

    Returns:
        Potential values on the grid (same shape as x and y).
    """
    V = 0.5 * omega_x**2 * (x - x0) ** 2 + 0.5 * omega_y**2 * (y - y0) ** 2
    return V


def potential_well_2d(
    x: np.ndarray,
    y: np.ndarray,
    well_depth: float = 10.0,
    well_width: float = 1.0,
    barrier_height: float = 20.0,
    perturbations: list = None,
) -> np.ndarray:
    """
    Compute 2D potential well with optional local perturbations.

    Base potential is a rectangular well. Perturbations can be added
    as localized Gaussian bumps or dips.

    Args:
        x: X-coordinate grid (2D array).
        y: Y-coordinate grid (2D array).
        well_depth: Depth of the potential well (negative value).
        well_width: Width of the well region.
        barrier_height: Height of the potential barrier outside the well.
        perturbations: List of perturbation dictionaries, each with:
            - 'x0': center x position
            - 'y0': center y position
            - 'amplitude': perturbation strength (positive for bump, negative for dip)
            - 'sigma': width of the Gaussian perturbation

    Returns:
        Potential values on the grid.
    """
    # Initialize with rectangular well
    V = np.zeros_like(x)

    # Inside well: negative potential
    inside_well = (np.abs(x) < well_width / 2) & (np.abs(y) < well_width / 2)
    V[inside_well] = -well_depth
    V[~inside_well] = barrier_height

    # Add perturbations
    if perturbations:
        for pert in perturbations:
            x0 = pert.get("x0", 0.0)
            y0 = pert.get("y0", 0.0)
            amplitude = pert.get("amplitude", 1.0)
            sigma = pert.get("sigma", 0.1)

            # Gaussian perturbation
            perturbation = amplitude * np.exp(
                -((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2)
            )
            V += perturbation

    return V


def build_hamiltonian_2d(
    V: np.ndarray,
    dx: float,
    dy: float,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> sparse.csr_matrix:
    """
    Build discrete Hamiltonian matrix for 2D quantum system.

    Uses finite difference approximation:
    H = T + V
    where T is kinetic energy operator and V is potential energy.

    Args:
        V: Potential values on 2D grid (shape: (Nx, Ny)).
        dx: Grid spacing in x direction.
        dy: Grid spacing in y direction.
        mass: Particle mass (in atomic units, default 1.0).
        hbar: Reduced Planck constant (in atomic units, default 1.0).

    Returns:
        Sparse Hamiltonian matrix (shape: (N, N) where N = Nx * Ny).
    """
    Nx, Ny = V.shape
    N = Nx * Ny

    # Flatten potential
    V_flat = V.flatten()

    # Kinetic energy coefficients
    coeff_x = -(hbar**2) / (2 * mass * dx**2)
    coeff_y = -(hbar**2) / (2 * mass * dy**2)

    # Build sparse matrix using diagonal format
    # Main diagonal: 2 * (coeff_x + coeff_y) + V
    diagonals = []
    offsets = []

    # Main diagonal (potential + kinetic diagonal term)
    main_diag = -2 * (coeff_x + coeff_y) * np.ones(N) + V_flat
    diagonals.append(main_diag)
    offsets.append(0)

    # Off-diagonals for x-direction coupling
    x_diag = coeff_x * np.ones(N - 1)
    # Zero out boundary crossings
    for i in range(Ny - 1):
        x_diag[Nx * (i + 1) - 1] = 0
    diagonals.append(x_diag)
    offsets.append(1)
    diagonals.append(x_diag)
    offsets.append(-1)

    # Off-diagonals for y-direction coupling
    y_diag = coeff_y * np.ones(N - Nx)
    diagonals.append(y_diag)
    offsets.append(Nx)
    diagonals.append(y_diag)
    offsets.append(-Nx)

    # Create sparse matrix
    H = sparse.diags(diagonals, offsets, shape=(N, N), format="csr")

    logger.debug(
        f"Built Hamiltonian matrix: shape {H.shape}, nnz={H.nnz}, "
        f"sparsity={1 - H.nnz / (N**2):.4f}"
    )

    return H


def create_grid_2d(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    Nx: int,
    Ny: int,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Create 2D coordinate grid for quantum system.

    Args:
        x_range: Tuple of (x_min, x_max).
        y_range: Tuple of (y_min, y_max).
        Nx: Number of grid points in x direction.
        Ny: Number of grid points in y direction.

    Returns:
        Tuple of (X, Y, dx, dy) where X and Y are 2D coordinate arrays,
        and dx, dy are grid spacings.
    """
    x = np.linspace(x_range[0], x_range[1], Nx)
    y = np.linspace(y_range[0], y_range[1], Ny)

    X, Y = np.meshgrid(x, y, indexing="ij")

    dx = (x_range[1] - x_range[0]) / (Nx - 1)
    dy = (y_range[1] - y_range[0]) / (Ny - 1)

    logger.info(
        f"Created 2D grid: {Nx}x{Ny} points, "
        f"x: [{x_range[0]}, {x_range[1]}], "
        f"y: [{y_range[0]}, {y_range[1]}], "
        f"dx={dx:.4f}, dy={dy:.4f}"
    )

    return X, Y, dx, dy
