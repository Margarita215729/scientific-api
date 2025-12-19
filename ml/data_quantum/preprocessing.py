"""
Preprocessing functions for quantum system data.

This module provides functions to generate quantum systems,
build Hamiltonians, and save results to disk.
"""

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
from scipy import sparse

from app.core.config import get_settings
from app.core.logging import get_logger
from ml.data_quantum.models import (
    build_hamiltonian_2d,
    create_grid_2d,
    harmonic_oscillator_2d_potential,
    potential_well_2d,
)

logger = get_logger(__name__)


def generate_harmonic_oscillator(
    x_range: tuple = (-3.0, 3.0),
    y_range: tuple = (-3.0, 3.0),
    Nx: int = 50,
    Ny: int = 50,
    omega_x: float = 1.0,
    omega_y: float = 1.0,
    x0: float = 0.0,
    y0: float = 0.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Generate 2D harmonic oscillator quantum system.

    Args:
        x_range: X coordinate range.
        y_range: Y coordinate range.
        Nx: Number of grid points in x.
        Ny: Number of grid points in y.
        omega_x: Angular frequency in x.
        omega_y: Angular frequency in y.
        x0: Center in x.
        y0: Center in y.
        mass: Particle mass.
        hbar: Reduced Planck constant.

    Returns:
        Dictionary with keys:
            - 'X': X coordinate grid
            - 'Y': Y coordinate grid
            - 'V': Potential values
            - 'H': Hamiltonian matrix (sparse)
            - 'dx': Grid spacing in x
            - 'dy': Grid spacing in y
    """
    logger.info(
        f"Generating harmonic oscillator: grid {Nx}x{Ny}, "
        f"omega_x={omega_x}, omega_y={omega_y}"
    )

    # Create grid
    X, Y, dx, dy = create_grid_2d(x_range, y_range, Nx, Ny)

    # Compute potential
    V = harmonic_oscillator_2d_potential(X, Y, omega_x, omega_y, x0, y0)

    # Build Hamiltonian
    H = build_hamiltonian_2d(V, dx, dy, mass, hbar)

    logger.info("Harmonic oscillator system generated successfully")

    return {
        "X": X,
        "Y": Y,
        "V": V,
        "H": H,
        "dx": dx,
        "dy": dy,
        "model_type": "harmonic_oscillator",
        "parameters": {
            "omega_x": omega_x,
            "omega_y": omega_y,
            "x0": x0,
            "y0": y0,
            "mass": mass,
            "hbar": hbar,
        },
    }


def generate_potential_well(
    x_range: tuple = (-2.0, 2.0),
    y_range: tuple = (-2.0, 2.0),
    Nx: int = 50,
    Ny: int = 50,
    well_depth: float = 10.0,
    well_width: float = 1.0,
    barrier_height: float = 20.0,
    perturbations: Optional[list] = None,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Generate 2D potential well quantum system with perturbations.

    Args:
        x_range: X coordinate range.
        y_range: Y coordinate range.
        Nx: Number of grid points in x.
        Ny: Number of grid points in y.
        well_depth: Depth of potential well.
        well_width: Width of the well.
        barrier_height: Height of barrier outside well.
        perturbations: List of perturbation dictionaries.
        mass: Particle mass.
        hbar: Reduced Planck constant.

    Returns:
        Dictionary with quantum system data.
    """
    logger.info(
        f"Generating potential well: grid {Nx}x{Ny}, "
        f"depth={well_depth}, width={well_width}, "
        f"perturbations={len(perturbations) if perturbations else 0}"
    )

    # Create grid
    X, Y, dx, dy = create_grid_2d(x_range, y_range, Nx, Ny)

    # Compute potential
    V = potential_well_2d(X, Y, well_depth, well_width, barrier_height, perturbations)

    # Build Hamiltonian
    H = build_hamiltonian_2d(V, dx, dy, mass, hbar)

    logger.info("Potential well system generated successfully")

    return {
        "X": X,
        "Y": Y,
        "V": V,
        "H": H,
        "dx": dx,
        "dy": dy,
        "model_type": "potential_well",
        "parameters": {
            "well_depth": well_depth,
            "well_width": well_width,
            "barrier_height": barrier_height,
            "perturbations": perturbations,
            "mass": mass,
            "hbar": hbar,
        },
    }


def save_quantum_system(
    system: Dict,
    output_path: Union[str, Path],
    save_hamiltonian: bool = True,
) -> None:
    """
    Save quantum system data to disk.

    Args:
        system: Dictionary with quantum system data.
        output_path: Path to save directory (without extension).
        save_hamiltonian: Whether to save the Hamiltonian matrix.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving quantum system to {output_path}")

    # Save grid coordinates and potential
    np.savez_compressed(
        f"{output_path}_grid.npz",
        X=system["X"],
        Y=system["Y"],
        V=system["V"],
        dx=system["dx"],
        dy=system["dy"],
        model_type=system["model_type"],
    )

    # Save Hamiltonian if requested
    if save_hamiltonian and "H" in system:
        sparse.save_npz(f"{output_path}_hamiltonian.npz", system["H"])

    # Save parameters
    import json

    with open(f"{output_path}_params.json", "w") as f:
        json.dump(system["parameters"], f, indent=2)

    logger.info(f"Quantum system saved to {output_path}")


def load_quantum_system(
    input_path: Union[str, Path],
    load_hamiltonian: bool = True,
) -> Dict:
    """
    Load quantum system data from disk.

    Args:
        input_path: Path to saved system (without extension).
        load_hamiltonian: Whether to load the Hamiltonian matrix.

    Returns:
        Dictionary with quantum system data.
    """
    input_path = Path(input_path)

    logger.info(f"Loading quantum system from {input_path}")

    # Load grid and potential
    grid_data = np.load(f"{input_path}_grid.npz", allow_pickle=True)

    system = {
        "X": grid_data["X"],
        "Y": grid_data["Y"],
        "V": grid_data["V"],
        "dx": float(grid_data["dx"]),
        "dy": float(grid_data["dy"]),
        "model_type": str(grid_data["model_type"]),
    }

    # Load Hamiltonian if requested
    if load_hamiltonian:
        hamiltonian_path = f"{input_path}_hamiltonian.npz"
        if Path(hamiltonian_path).exists():
            system["H"] = sparse.load_npz(hamiltonian_path)

    # Load parameters
    import json

    params_path = f"{input_path}_params.json"
    if Path(params_path).exists():
        with open(params_path, "r") as f:
            system["parameters"] = json.load(f)

    logger.info(f"Quantum system loaded from {input_path}")
    return system
