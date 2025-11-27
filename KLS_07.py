"""

N O R T H   S U L A W E S I   T S U N A M I  
configuration : KL slip 2D
===========================================

"""


import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from numpy import random
from scipy import stats
from scipy.stats import gaussian_kde
from tqdm import tqdm

from clawpack.geoclaw import dtopotools, topotools
from clawpack.visclaw import colormaps

import warnings
import pickle


# ======================================================================= 
# Global Parameters
# =======================================================================

# topo. dtopo and shorelines file names
TOPO_FILENAME: str = 'sulawesiP116N6P130P8.asc'
SHORE_FILENAME: str = "sulawesiP116N6P130P8.npy"
FAULT_FILENAME_SELECTED: str = "sulawesi_selected.csv"
FAULT_FILENAME: str = "sulawesi.csv"

# max dip of sulawesi faults
MAX_DEPTH: float = 71000.0
    
# realization test and generation parameters
NTRIAL_TEST: int = 10000                 # 10000
NTERM_TEST: int = 60
NTRIAL_GEN: int = 20000                   # 20000
NTERM_GEN_LIST: List[int] = [7, 60]            # only two is allowed for graph
MODES: Tuple[int, int] = (2, 4)

# set directory filenames
KLS_DIR: str = os.environ.get("KLS_DIR", "KLS_07_output")
os.makedirs(KLS_DIR, exist_ok=True)
TOPO_DIR: str = os.environ.get("TOPO_DIR", "topo")

# scenario parameters
USE_SELECTED_SCENARIOS: bool = True
SCENARIO_DIR: str = os.environ.get("SCENARIO_DIR", "scenario")
PTS_FILENAME: str = os.path.join(SCENARIO_DIR, "scenario_pts_final.txt")
WGTS_FILENAME: str = os.path.join(SCENARIO_DIR, "scenario_prb_wgts_final.txt")

# Fault parameters
NDIP: int = 2        # dip refinement factor
NSTRIKE: int = 2     # strike refinement factor
MW_DESIRED: float = 8.8            # check Jaja
DIP_RANGE: Tuple[int, int] = (0, 9)          # check Jaja
STRIKE_RANGE: Tuple[int, int] = (0, 30)     # check Jaja

# Set Point of Interest (POI) and Area of Interest (AOI), make sure POI is in AOI
POI: Tuple[float, float] = (121.25, 1.236)
AOI: Dict[str, Any] = {
    "x": (119.0, 122.5),
    "y": (0.4, 2.25),
    "nx": 352,
    "ny": 170,
}
YLIM_POI: List[float] = [0.4, 2.25]

# realizations file name
REALIZATION_PKL_FILENAME: str = "realizations_dict.pkl"

# contour interval of deformation level contour
DZ_INTERVAL: float = 1.0

# figure outputs file names:
SUBFAULT_DIVIDED_FILENAME: str  = "sulawesi_subfault_divided.png"
HAZARD_CURVE_FILENAME: str      = "hazard_curve.png"
DB_SHORE_CURVE_FILENAME: str    = "db_shore_sample"
KDE_ETAMAX_FILENAME: str        = "kde_test_01.png"
KDE_UPLIFT_FILENAME: str        = "kde_test_02.png"
EIGEN_MODES_FILENAME: str       = "CSZmodes.png"
EIGENVALUES_FILENAME: str       = "eigenvalues.png"
REALIZATION_FILENAME: str       = "CSZrealizations.png"



# helper to save figures
def savefigp(info: str = "[INFO]:", fname: str = 'fig.png', fig: Optional[Figure] = None) -> None:
    """Save a figure to the output directory, defaulting to the current Matplotlib figure."""
    subdir = KLS_DIR
    figure = fig or plt.gcf()
    fname = os.path.join(subdir, fname)
    figure.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"{info} > File name: {fname}")
    plt.close(figure)

# --- end Global Parameters  and functions


# (1) Setup Fault ---------------------------------------------------------------------------------

# 1.1. Read and display topography 
def setup_topo(topo_filename: str, topo_type: int = 2, make_plot: Optional[bool] = None) -> topotools.Topography:
    """read and display topography"""
    topo = topotools.Topography(topo_filename, topo_type=2)
    print(f"reading topography: {topo_filename}, topo type = {topo_type}")
    topo.read(topo_filename, topo_type=topo_type)

    if make_plot:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.contourf(topo.X, topo.Y, topo.Z, [0, 20000], colors=[[.3, 1, .3]])
        info = "[INFO  1.1]: Saving Topography"
        savefigp(info=info, fname="topography.png", fig=fig)

    return topo


# 1.2. Fault Geometry Computation
def compute_fault_geometry(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Compute:
      - selected_strike_length : total length along strike direction (top depth row)
      - selected_dip_length    : mean dip length across columns
      - total_area    : sum of (Length * Width) of all subfaults
    """
    df = df.reset_index(drop=True)

    depths = np.sort(df["Depth (km)"].unique())
    ndip = len(depths)
    nstrike = len(df) // ndip

    # Strike length (top row)
    top_depth = depths[0]
    strike_length = df[df["Depth (km)"] == top_depth]["Length"].sum()

    # Dip length (mean across columns)
    dip_lengths = []
    for j in range(nstrike):
        idx = [i * nstrike + j for i in range(ndip)]
        dip_lengths.append(df.loc[idx, "Width"].sum())

    dip_length = float(np.mean(dip_lengths))

    # Total area
    total_area = float((df["Length"] * df["Width"]).sum())

    print("[INFO  1.2]: Full fault geometry:")
    print(f"  Strike length     : {strike_length:.3f} km")
    print(f"  Dip length        : {dip_length:.3f} km")
    print(f"  Total area        : {total_area:.3f} km²")
    print(f"  Number of Strike  : {nstrike}")
    print(f"  Number of dip     : {ndip}")

    return strike_length, dip_length


# 1.3. select subfault and compute its geometry
def select_fault(df: pd.DataFrame,
                 dip_range: Tuple[int, int],
                 strike_range: Tuple[int, int],
                 output_path: str = "selected_fault.csv") -> Tuple[pd.DataFrame, float, float]:
    """
    Select a subset of the fault model based on dip and strike indices,
    save result to CSV with the same format as the input,
    and compute geometry for the selected subset.
    Returns:
        df_selected, selected_strike_length, selected_dip_length
    """
    df = df.reset_index(drop=True)
    depths = np.sort(df["Depth (km)"].unique())
    ndip = len(depths)
    nstrike = len(df) // ndip

    dip_min, dip_max = dip_range
    strike_min, strike_max = strike_range

    # Build dip/strike index arrays
    idx_global = np.arange(len(df))
    dip_idx = idx_global // nstrike
    strike_idx = idx_global % nstrike

    df_tmp = df.copy()
    df_tmp["dip_idx"] = dip_idx
    df_tmp["strike_idx"] = strike_idx

    # Apply range filtering
    mask = (
        (df_tmp["dip_idx"] >= dip_min) & (df_tmp["dip_idx"] <= dip_max) &
        (df_tmp["strike_idx"] >= strike_min) & (df_tmp["strike_idx"] <= strike_max)
    )

    df_selected = df_tmp[mask].drop(columns=["dip_idx", "strike_idx"])

    # Save to CSV
    df_selected.to_csv(output_path, index=False)

    # Compute geometry for selected fault
    strike_length_selected, dip_length_selected = compute_fault_geometry(df_selected)

    # Total area
    total_area = float((df["Length"] * df["Width"]).sum())

    print("\n[INFO  1.3]: Selected fault geometry:")
    print(f"- Strike length     : {strike_length_selected:.3f} km")
    print(f"- Dip length        : {dip_length_selected:.3f} km")
    print(f"- Total area    :    {total_area:.3f} km²")
    print(f"- Number of Strike  : {nstrike}")
    print(f"- Number of dip     : {ndip}")

    return df_selected, strike_length_selected, dip_length_selected

# 1.4. setup fault
def setup_fault(subfault_fname: str) -> dtopotools.Fault:
    """Read fault parameters from sulawesi.csv and return relevant values."""
    column_map = {"longitude": 1, "latitude": 2, "depth": 3, "strike": 4,
                  "length": 5, "width": 6, "dip": 7}
    defaults = {'rake': 90, 'slip': 1.0}
    coordinate_specification = 'top center'
    input_units = {'slip': 'm', 'depth': 'km', 'length': 'km', 'width': 'km'}
    rupture_type = 'static'
    skiprows = 1
    delimiter = ','
    fault = dtopotools.CSVFault()
    fault.read(subfault_fname, 
               input_units=input_units, 
               coordinate_specification="top center",
    )
    print (f"[INFO  1.4]: ORIGINAL FAULT from Fault Geometry: {subfault_fname}:")
    print (f"  There are {len(fault.subfaults)} subfaults")

    # return fault in meter by dtopotools
    return fault

# 1.5. Subdivide the fault into smaller subfaults
def subdivide_fault(fault: dtopotools.Fault, nstrike: int, ndip: int) -> dtopotools.Fault:
    """Subdivide the fault into smaller subfaults for 2D analysis."""
    PHI_PLATE = 60.         # angle oceanic plate moves clockwise from north, to set rake
    new_subfaults = []      # initialize subfault matrix
    subfault_iter = tqdm(
        fault.subfaults,
        total=len(fault.subfaults),
        desc="Subdividing fault",
        unit="subfault",
    )
    for subfault in subfault_iter:
        subfault.rake = subfault.strike - PHI_PLATE - 180.
        f = dtopotools.SubdividedPlaneFault(subfault, nstrike, ndip)
        new_subfaults += f.subfaults
    
    # reset fault.subfaults to the new list of all subfaults after subdividing:
    new_fault = dtopotools.Fault(subfaults=new_subfaults)
    n = len(new_fault.subfaults)
    print (f"[INFO  1.5]: Subdivided selected fault")
    print (f"- Subdivision scale : {nstrike} strike scale x {ndip} dip scale")
    print (f"- Subdivided fault has {n} subfaults")

    return new_fault

# 1.6. plot sub fault and new sub faul
def plot_subfault(fault_org: dtopotools.Fault,
                  fault_selected: dtopotools.Fault,
                  fault_divided: dtopotools.Fault,
                  shore: np.ndarray,
                  filename: str = 'sub_fault.png') -> None:
    """plot sub fault and new sub fault"""

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # local plot helper
    def plot_fault(axs, fault, line_color="orange", line_width=1.0):
        line_offset = len(axs.lines)            
        fault.plot_subfaults(axs)               
        for line in axs.lines[line_offset:]:    
            line.set_color(line_color)
            line.set_linewidth(line_width)

    # plot selected fault
    axs[0].plot(shore[:,0], shore[:,1], 'g', linewidth=3)
    plot_fault(axs[0], fault_org, line_color="gray", line_width=2)
    plot_fault(axs[0], fault_selected, line_color="red", line_width=0.5)
    axs[0].axis([119, 125, -0.5, 2.5])
    axs[0].set_title('Selected Fault')      

    # plot new divided fault
    axs[1].plot(shore[:,0], shore[:,1], 'g', linewidth=2)
    plot_fault(axs[1], fault_org, line_color="gray", line_width=2)
    plot_fault(axs[1], fault_divided, line_color="red", line_width=0.5)
    axs[1].axis([119, 125, -0.5, 2.5] )
    axs[1].set_title('Divided Fault')

    for k in range(2):
        axs[k].xaxis.set_ticks_position('both')
        axs[k].yaxis.set_ticks_position('both')
        axs[k].tick_params(axis='both', which='both', direction='in', top=True, right=True)
        axs[k].xaxis.set_major_locator(MaxNLocator(6))
        axs[k].yaxis.set_major_locator(MaxNLocator(6))

    info = "[INFO  1.6]: Plot sub fault and new sub fault"
    savefigp(info=info, fname=filename, fig=fig)

# Interpolate POI at dtopo area
def interpolate_point_of_interest(
    dZr: Optional[np.ndarray] = None,
    dtopo: Optional[dtopotools.DTopography] = None,
) -> Union[bool, float]:
    """
    When called with `dtopo`, compute and store interpolation weights for the POI,
    returning True if the point lies within bounds. When called with `dZr`,
    return the interpolated displacement using the stored weights.
    """
    cache = getattr(interpolate_point_of_interest, "_cache", None)

    if dtopo is not None:
        xcc, ycc = POI
        i1cc = np.where(dtopo.x < xcc)[0].max()
        j1cc = np.where(dtopo.y < ycc)[0].max()
        a1cc = (xcc - dtopo.x[i1cc]) / (dtopo.x[i1cc + 1] - dtopo.x[i1cc])
        a2cc = (ycc - dtopo.y[j1cc]) / (dtopo.y[j1cc + 1] - dtopo.y[j1cc])
        cache = {
            "i1cc": i1cc,
            "j1cc": j1cc,
            "a1cc": a1cc,
            "a2cc": a2cc,
            "valid": (0.0 <= a1cc <= 1.0) and (0.0 <= a2cc <= 1.0),
        }
        interpolate_point_of_interest._cache = cache
        if not cache["valid"]:
            print('*** Interpolation to CC not correct!')
        return cache["valid"]

    if cache is None:
        raise RuntimeError(
            "interpolate_point_of_interest requires initialization with dtopo first."
        )

    i1cc, j1cc, a1cc, a2cc = (
        cache["i1cc"],
        cache["j1cc"],
        cache["a1cc"],
        cache["a2cc"],
    )
    valid = cache["valid"]
    if dZr is None:
        if not valid:
            print('*** Interpolation to CC not correct!')
        return valid

    if not valid:
        warnings.warn("POI interpolation weights invalid; returning NaN.")
        return np.nan

    dzy1 = (1. - a1cc) * dZr[j1cc, i1cc] + a1cc * dZr[j1cc, i1cc + 1]
    dzy2 = (1. - a1cc) * dZr[j1cc + 1, i1cc] + a1cc * dZr[j1cc + 1, i1cc + 1]
    dzcc = (1. - a2cc) * dzy2 + a2cc * dzy1
    return dzcc


# (2) KL Expansion --------------------------------------------------------------------------------


# 2.1. Compute desired M0
def fix_M0_desired(Mw_desired: float) -> float:
    return 10.**(1.5 * Mw_desired + 9.05)

# 2.2. Estimate distances between subfaults in strike and dip directions
def compute_subfault_distances(fault: dtopotools.Fault) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate distances between subfaults in strike and dip directions."""
    rad = np.pi / 180.
    rr = 6.378e6  # Earth radius (m)
    lat2meter = rr * rad # latitude to meters
    MIN_SIN = 1e-3 # to avoid Ddip > Depth
    
    nsubfaults = len(fault.subfaults)
    D = np.zeros((nsubfaults, nsubfaults))
    Dstrike = np.zeros((nsubfaults, nsubfaults))
    Ddip = np.zeros((nsubfaults, nsubfaults))    
    subfault_iter = tqdm(
        fault.subfaults,
        total=nsubfaults,
        desc="Computing subfault distances",
        unit="subfault",
    )
    for i, si in enumerate(subfault_iter):
        xi, yi, zi = si.longitude, si.latitude, si.depth
        for j, sj in enumerate(fault.subfaults):
            xj, yj, zj = sj.longitude, sj.latitude, sj.depth
            dx = abs(xi-xj) * np.cos(0.5*(yi+yj)*np.pi/180.) * lat2meter
            dy = abs(yi-yj) * lat2meter
            dz = abs(zi-zj)
            # Euclidean distance:
            D[i, j] = np.sqrt(dx**2 + dy**2 + dz**2) 
            # estimate distance down-dip based on depths:
            dip = 0.5 * (si.dip + sj.dip)
            sin_dip = np.maximum(np.sin(dip * np.pi / 180.), MIN_SIN)
            ddip1 = dz / sin_dip
            Ddip[i, j] = ddip1
            Ddip[i, j] = min(ddip1, D[i, j]) # 29/08/2025
            dstrike2 = max(D[i, j]**2 - Ddip[i, j]**2, 0.0)
            Dstrike[i, j] = np.sqrt(dstrike2)
    
    return D, Dstrike, Ddip

# 2.3. Compute 2D correlation matrix
def compute_correlation_matrix(
    Dstrike: np.ndarray,
    Ddip: np.ndarray,
    selected_strike_length: float,
    selected_dip_length: float,
) -> np.ndarray:
    """Compute 2D correlation matrix."""
    r = np.sqrt((Dstrike / selected_strike_length)**2 + (Ddip / selected_dip_length)**2)
    C = np.exp(-r)
    
    return C

# 2.4. Compute mean slip
def compute_mean_slip(fault: dtopotools.Fault, M0_desired: float) -> float:
    """Compute mean slip from desired Mw """
    lengths = np.array([s.length for s in fault.subfaults])
    widths = np.array([s.width for s in fault.subfaults])
    areas = lengths * widths
    total_area = sum(areas)

    mean_slip = M0_desired / (fault.subfaults[0].mu * total_area)

    return mean_slip 

# 2.5. Find eigenvalues, and eigenvector matrix. Columns V[:,k] are eigenvectors.
def compute_eigenmodes(C: np.ndarray, mean_slip: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute eigenvalues and eigenvectors of the covariance matrix."""
    alpha = 0.5
    sigma_slip = alpha * mean_slip*np.ones_like(C)
    # Lognormal:
    Cov_g = np.log((sigma_slip/mean_slip) * (C*(sigma_slip/mean_slip)).T + 1.)
    mean_slip_g = np.log(mean_slip) - np.diag(Cov_g)/2.
    # This should be the same:
    Cov_g_1 = Cov_g
    Cov_g = np.log(alpha**2 * C + 1.)
    # mean_slip_g = np.log(mean_slip) - np.diag(Cov_g) / 2.
    absolute_difference = np.linalg.norm(Cov_g_1 - Cov_g, ord='fro')
    print(f"- Absolute difference (Frobenius norm): {absolute_difference}")
    print(f"- Finding eigenmodes from matrix C ({C.shape})")
    lam, V = np.linalg.eig(Cov_g)
    # takes real parts only
    lam = np.real(lam)
    V = np.real(V)
    # sort eigen values and vectors
    i = list(np.argsort(lam))
    i.reverse()
    lam = lam[i]
    V = V[:, i]
    
    return lam, V, mean_slip_g

# 2.6. Plot eigen values 
def plot_eigenvalues(lam: np.ndarray, kplot: int = 20, filename: str = EIGENVALUES_FILENAME) -> None:
    """Plot the first kplot eigenvalues."""
    fig, ax = plt.subplots(figsize=(5, 4))
    k = np.arange(1, kplot + 1)
    lam_slice = lam[:kplot]
    ax.loglog(k, lam_slice, 'o', label=r'$\lambda$')
    numer = float(lam[0])
    ax.plot(k, numer / k**2, 'r', label=f'${numer:.1f}/k^2$')

    # Fit log10(lambda) ~ m log10(k) + b to visualize decay rate.
    positive_mask = lam_slice > 0
    if np.count_nonzero(positive_mask) >= 2:
        logk = np.log10(k[positive_mask])
        loglam = np.log10(lam_slice[positive_mask])
        slope, intercept = np.polyfit(logk, loglam, 1)
        amplitude = 10 ** intercept
        fit_curve = amplitude * k ** slope
        ax.loglog(k, fit_curve, '--', color='gray',
            label=rf'fit: ${amplitude:.1f}/k^{{{np.abs(slope):.2f}}}$')

    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='both', labelsize=12, direction='in', top=True, right=True)
    ax.set_title(f"First {kplot} eigen values of covariance matrix", fontsize=12)
    ax.set_xlabel('Mode number k', fontsize=12)
    ax.set_ylabel('Eigenvalue λ', fontsize=12)

    info = "[INFO  2.6]: Saving eigen values"
    savefigp(info, filename, fig=fig)


# 2.7. Plot eigen modes
def plot_eigenmodes(new_fault: dtopotools.Fault,
                    V: np.ndarray,
                    modes: Tuple[int, int],
                    slip_factor: float = 15.,
                    filename: str = 'CSZmodes.png') -> None:
    """Plot the first `modes[0] * modes[1]` eigenmodes on a regular grid."""

    nrows, ncols = modes
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
    # Ensure axes is a 2D numpy array with shape (nrows, ncols).
    # plt.subplots may return a single Axes instance, a 1D array, or a 2D array.
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        axes = axes.reshape(nrows, ncols)
    cmap_slip = colormaps.make_colormap({0: 'g', 0.5: 'w', 1.: 'm'})
    total = min(nrows * ncols, V.shape[1])

    for idx in range(total):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        for j, s in enumerate(new_fault.subfaults):
            s.slip = -V[j, idx] * slip_factor

        new_fault.plot_subfaults(
            ax,
            slip_color=True,
            cmin_slip=-1,
            cmax_slip=1,
            plot_box=0,
            cmap_slip=cmap_slip,
            colorbar_shrink=0,
        )
        ax.set_title(f"Mode {idx}", fontsize=18)
        ax.axis('off')

    # Hide any unused axes
    for idx in range(total, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis('off')
    fig.tight_layout()
    
    info = "[INFO  2.6]: Saving eigen values"
    savefigp(info, filename, fig=fig)

# Formulate taper function
def get_taper_function(
    taper_name: str,
    x: np.ndarray,
    depth0: float = 10e3,
    MAX_DEPTH: float = 32500.,
    dip: float = 13.,
    W: float = 100e3,
    make_plot: Optional[bool] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return the specified taper function using a dictionary."""
    # none (no taper applied)
    def none(x):
        tau = np.ones(x.shape)
        return tau
    
    # cospower
    def cospower(x):
        power = 10
        tau =  1. - (1. - 0.5 * (1. + np.cos(2 * np.pi * (x + 0.5 * W) / W)))**power
        return tau
    
    # cospower downdip
    def cospower_downdip(x):
        power = 10
        tau =  np.where(x < W / 2, 1., 1. - (1. - 0.5 * (1. + np.cos(2 * np.pi * (x + 0.5 * W) / W)))**power)
        return tau

    # WangHe
    def WangHe(x):
        broadness = 0.25
        qskew = 0.65

        def delta(xp):
            d1 = (12. / qskew**3) * xp**2 * (qskew / 2. - xp / 3.)
            dq = 2.
            d2 = dq + (12. / (1. - qskew)**3) * (
                (xp**3 / 3. - xp**2 * (1 + qskew) / 2. + qskew * xp) -
                (qskew**3 / 3. - qskew**2 * (1 + qskew) / 2. + qskew**2))
            return np.where(xp <= qskew, d1, d2)
        
        def tau_x(x):
            xp = (x - x[0]) / (x[-1] - x[0])
            return delta(xp) * (1. + np.sin(broadness * np.pi * delta(xp)))
        
        tau = tau_x(x)
        return tau
    
    # exponetial depth
    def exp_depth(x):
        tau_depth = lambda d: 1. - np.exp((d - MAX_DEPTH) * 20 / MAX_DEPTH)
        tau = tau_depth(depth0 + x * np.sin(dip * np.pi / 180.))
        return tau
    
    taper_functions = {
        'none': none,
        'cospower': cospower,
        'cospower_downdip': cospower_downdip,
        'WangHe': WangHe,
        'exp_depth': exp_depth
    }
    
    if taper_name not in taper_functions:
        raise ValueError(f"Unknown taper: {taper_name}. Choose from {list(taper_functions.keys())}")
    print(f"Selected taper function: {taper_name}")

    # Buat plot jika make_plot=True atau (make_plot=None dan taper='exp_depth')
    if make_plot or (make_plot is None and taper_name == 'exp_depth'):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(x/1000., tau(x), label=f'Taper: {taper_name}')
        ax.set_xlabel('km down-dip')
        ax.set_ylabel('Taper value')
        ax.set_title('Taper Function')
        ax.set_ylim(-0.1, 1.1)
        ax.legend()
        ax.grid(True)
        savefigp(info="[INFO]: Taper function", fname='taper.png', fig=fig)

    return taper_functions[taper_name]

# Compute sea floor deformation for each subfault
def compute_deformation(
    fault: dtopotools.Fault,
    x_dtopo: np.ndarray,
    y_dtopo: np.ndarray,
) -> Tuple[np.ndarray, dtopotools.DTopography]:
    """Compute sea floor deformation for each subfault."""
    n_subfaults = len(fault.subfaults)
    dZ = np.zeros((len(y_dtopo), len(x_dtopo), n_subfaults))
    print(f'  Computing deformation for {n_subfaults} subfault')
    for j in tqdm(range(n_subfaults), desc="Deformation", unit="subfault"):
        sfault = dtopotools.Fault(subfaults=[fault.subfaults[j]])
        sfault.subfaults[0].slip = 1.
        dtopo = sfault.create_dtopography(x_dtopo, y_dtopo, times=[1.], verbose=False)
        dZ[:, :, j] = dtopo.dZ[0, :, :]

    return dZ, dtopo

# 2.9. Compute sea floor deformation for each subfault at AOI
def compute_deformation_AOI(
    fault: dtopotools.Fault,
    AOI: Dict[str, Any],
) -> Tuple[np.ndarray, dtopotools.DTopography, np.ndarray, np.ndarray]:
    """
    Compute deformation for an AOI defined by lon/lat bounds and grid sizes.
    """
    x_bounds = AOI["x"]
    y_bounds = AOI["y"] 
    nx = AOI["nx"] 
    ny = AOI["ny"]
    x_dtopo = np.linspace(x_bounds[0], x_bounds[1], nx)
    y_dtopo = np.linspace(y_bounds[0], y_bounds[1], ny)
    dZ, dtopo = compute_deformation(fault, x_dtopo, y_dtopo)

    return dZ, dtopo, x_dtopo, y_dtopo


# KL expansion
def KL_expansion(
    z: np.ndarray,
    lam: np.ndarray,
    V: np.ndarray,
    new_fault: dtopotools.Fault,
    tau: Callable[[float], float],
    M0_desired: float,
) -> np.ndarray:
    """Perform 2D K-L expansion to compute slip."""
    nsubfaults = len(new_fault.subfaults)
    KL_slip = np.zeros(nsubfaults)
    # add in the terms in the K-L expansion:  (dropping V[:,0]) 
    for k in range(1, len(z)):
        KL_slip += z[k] * np.sqrt(lam[k]) * V[:, k]
    # Lognormal:
    KL_slip = np.exp(KL_slip)
    # Set the fault slip for the resulting realization:
    for j, s in enumerate(new_fault.subfaults):
        s.slip = KL_slip[j] * tau(s.depth)
    # Rescale to have desired magnitude:
    Mo = new_fault.Mo()
    KL_slip *= M0_desired / Mo
    for j, s in enumerate(new_fault.subfaults):
        s.slip = KL_slip[j] * tau(s.depth)
    
    return KL_slip

# Compute potensial energy
def compute_PotentialEnergy(dZr: np.ndarray, y_dtopo: np.ndarray) -> float:
    """Compute Potential Energy"""
    dy = 1./60. * 111.e3  # m
    # Use y_dtopo grid for correct shape
    dx = dy * np.cos(y_dtopo[:, None] * np.pi / 180.)  # shape (361, 1), broadcasts to (361, 181)
    grav = 9.81  # m/s^2
    rho_water = 1000  # kg/m^3
    # Mask land using topo.Z interpolated to dZr grid, or just mask nothing if dZr is only ocean
    # If you want to mask land, you need topo.Z interpolated to dZr grid. Otherwise, skip masking:
    # eta = ma.masked_where(interpolated_topoZ > 0, dZr)
    eta = dZr  # if dZr is already only ocean
    Energy = np.sum(eta**2 * dx * dy) * grav * rho_water * 1e-15  # PetaJoules

    return float(Energy)

# 2.10 Test realizations
def test_realizations(
    ntrials: int,
    nterms: int,
    dZ: np.ndarray,
    y_dtopo: np.ndarray,
    fault_selected: dtopotools.Fault,
    lam: np.ndarray,
    V: np.ndarray,
    tau: Callable[[float], float],
    M0_desired: float,
) -> pd.DataFrame:
    """Generate realizations and compute quantities.

    Modification:
      - If files "scenario_pts_final.txt" and "scenario_prb_wgts_final.txt" exist,
        use those scenarios (rows as z[1:], with z[0]=0 prepended) instead of random.randn().
      - Otherwise, behave exactly as before.
    """

    Energy = np.zeros(ntrials)
    Amplitude = np.zeros(ntrials)
    z_shore = np.zeros(ntrials)
    EtaMax = np.zeros(ntrials)
    zvals = np.zeros((ntrials, nterms + 1))

    # --- check for preselected scenarios ---
    pts_file = PTS_FILENAME
    wgt_file = WGTS_FILENAME
    use_selected = os.path.exists(pts_file) and os.path.exists(wgt_file)

    # Ensure these variables are always defined to avoid static-analysis / runtime issues
    Z_in = None
    W_in = None
    m_eff = 0

    if use_selected:
        print(f"\nLoading selected scenarios from {pts_file} and {wgt_file}")
        Z_in = np.loadtxt(pts_file)
        W_in = np.loadtxt(wgt_file)
        if Z_in.ndim == 1:
            Z_in = Z_in[None, :]
        nsel, dsel = Z_in.shape
        m_eff = min(dsel, nterms)
        ntrials = nsel  # override number of trials
        print(f"Using {ntrials} preselected scenarios ({m_eff} KL terms).")
        print(f"  scenario file point       : {pts_file}")
        print(f"  scenario file probability : {wgt_file}")
    else:
        print(f"\nTest realizations: {ntrials} trials for {nterms} terms")

    for j in tqdm(range(ntrials), desc="Realizations", unit="trial"):
        if use_selected:
            # Build z vector with 0th mode padded
            z = np.zeros(nterms + 1)
            z[1: m_eff + 1] = Z_in[j, :m_eff]
        else:
            z = random.randn(nterms + 1)

        zvals[j, :] = z
        KL_slip = KL_expansion(z, lam, V, fault_selected, tau, M0_desired)
        dZr = np.dot(dZ, KL_slip)  # linear combination of dZ from unit sources
        Energy[j] = compute_PotentialEnergy(dZr, y_dtopo)
        # z_offshore = where(topo.Z < 0, dZr, 0.)
        z_offshore = dZr.copy()
        Amplitude[j] = z_offshore.max() - z_offshore.min()
        z_shore[j] = interpolate_point_of_interest(dZr=dZr)
        EtaMax[j] = z_offshore.max()

    # save into pandas data frame
    realizations = pd.DataFrame({
        'Energy': Energy,
        'uplift': z_shore,
        'EtaMax': EtaMax,
        'depth proxy': EtaMax - z_shore,
    })

    return realizations

# 2.11. Plot sample realizations for different numbers of K-L terms
def plot_realizations(
    fault_selected: dtopotools.Fault,
    dtopo: dtopotools.DTopography,
    dZ: np.ndarray,
    y_dtopo: np.ndarray,
    lam: np.ndarray,
    V: np.ndarray,
    tau: Callable[[float], float],
    nterms: int,
    M0_desired: float,
    filename: str,
) -> None:
    """Plot sample realizations for different numbers of K-L terms:
      - If files "scenario_pts_final.txt" and "scenario_prb_wgts_final.txt" exist,
        use those scenarios (rows as z[1:], with z[0]=0 if needed) instead of random.randn().
      - Otherwise, behave exactly as before.
    """
    NTERM_REDUCED = 7
    random.seed(13579)

    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    axes_flat = axes.flatten()

    # --- Check if selected scenarios exist ---
    use_selected = False
    if USE_SELECTED_SCENARIOS:
        pts_file = PTS_FILENAME
        wgt_file = WGTS_FILENAME
        use_selected = os.path.exists(pts_file) and os.path.exists(wgt_file)

    nshow = 0
    m_eff = nterms  # Initialize m_eff with default value
    Z_in = None  # Initialize Z_in
    if use_selected:
        print(f"\nLoading selected scenarios from {pts_file} and {wgt_file}")
        Z_in = np.loadtxt(pts_file)
        W_in = np.loadtxt(wgt_file)
        if Z_in.ndim == 1:
            Z_in = Z_in[None, :]
        nsel, dsel = Z_in.shape
        m_eff = min(dsel, nterms)
        nshow = min(5, nsel)
        print(f"Using first {nshow} preselected scenarios for plotting.")
    else:
        print(f"\nPlot sample realizations: random {nterms}-term examples")

    for i in range(1, 6):
        if use_selected and i <= nshow and Z_in is not None:
            z = np.zeros(nterms)
            z[:m_eff] = Z_in[i - 1, :m_eff]
        else:
            z = np.random.randn(nterms)

        # --- Full nterms case ---
        KL_slip = KL_expansion(z, lam, V, fault_selected, tau, M0_desired)
        ax = axes_flat[i - 1]
        fault_selected.plot_subfaults(ax, slip_color=True, cmax_slip=20.,
                                 plot_box=False, colorbar_shrink=0)
        ax.axis('off')
        ax.set_title('Realization %i\n %i terms' % (i, nterms), fontsize=14)

        dz_max = np.maximum(abs(dtopo.dZ).max(), 0.8)

        dZr = np.dot(dZ, KL_slip)  # linear combination of dZ from unit sources
        ax = axes_flat[5 + i - 1]
        dtopotools.plot_dZ_colors(dtopo.X, dtopo.Y, dZr, axes=ax,
                                  cmax_dZ=dz_max, dZ_interval=DZ_INTERVAL, add_colorbar=False)
        
        ax.set_ylim(YLIM_POI)
        ax.plot(POI[0], POI[1], 'wo')
        ax.plot(POI[0], POI[1], 'kx')
        Energy = compute_PotentialEnergy(dZr, y_dtopo)
        dz_poi = interpolate_point_of_interest(dZr=dZr)
        ax.set_title('E=%4.2f,\n dB=%5.2f' % (Energy, dz_poi), fontsize=14)
        ax.axis('off')

        # --- Reduced nterms case ---
        z2 = z[:NTERM_REDUCED]
        KL_slip = KL_expansion(z2, lam, V, fault_selected, tau, M0_desired)
        ax = axes_flat[10 + i - 1]
        fault_selected.plot_subfaults(ax, slip_color=True, cmax_slip=20.,
                                 plot_box=False, colorbar_shrink=0)
        ax.axis('off')
        ax.set_title('%i terms' % NTERM_REDUCED, fontsize=14)


        dZr = np.dot(dZ, KL_slip)
        ax = axes_flat[15 + i - 1]

        dtopotools.plot_dZ_colors(dtopo.X, dtopo.Y, dZr, axes=ax,
                                  cmax_dZ=dz_max, dZ_interval=DZ_INTERVAL, add_colorbar=False)
        ax.set_ylim(YLIM_POI)
        ax.plot([POI[0]], [POI[1]], 'wo')
        ax.plot([POI[0]], [POI[1]], 'kx')
        Energy = compute_PotentialEnergy(dZr, y_dtopo)
        dz_poi = interpolate_point_of_interest(dZr=dZr)
        ax.set_title('E=%4.2f,\n dB=%5.2f' % (Energy, dz_poi), fontsize=14)
        ax.axis('off')

    info = "[INFO 2.11]: Saving realizations"
    fig.tight_layout()
    savefigp(info, filename, fig=fig)

# 2.12. Generate realizations (PASS)

# 2.13.  save realizatioN (PASS)

# 2.14. Plot Hazard curve
def plot_hazard_curves(realizations_dict: Dict[int, pd.DataFrame], zetai: np.ndarray, filename: str) -> None:
    """Plot hazard curves for depth proxy."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for nterms, realizations in realizations_dict.items():
        counts = np.zeros(len(zetai))
        for j in range(len(zetai)):
            counts[j] = np.sum(realizations['depth proxy'] > zetai[j])
        prob = counts / len(realizations)
        ax.semilogy(zetai, prob, label=f'{nterms} terms')
    
    ax.legend(loc='lower left', fontsize=16)
    ax.set_title('Hazard curve for depth proxy', fontsize=18)
    ax.set_xlabel('Exceedance value (meters)', fontsize=16)
    ax.set_ylabel('probability', fontsize=16)
    ax.tick_params(labelsize=16)

    savefigp(info="[INFO 2.14]: Hazard curve", fname=filename, fig=fig)

# 
def plot_kde(
    realizations_dict: Dict[int, pd.DataFrame],
    dzs: np.ndarray,
    nterms_list: Sequence[int] = (60, 7),
    filename: str = "db_shore_sample.png",
) -> None:
    """Plot kernel density estimates for subsidence/uplift."""
    fig, ax = plt.subplots(figsize=(5, 3))
    for nterms in nterms_list:
        kde = stats.gaussian_kde(realizations_dict[nterms]['uplift'])
        rho = kde.pdf(dzs)
        ax.plot(dzs, rho, label=f'{nterms} terms')

    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_title("Kernel density estimates for dZ at shore", fontsize=14)
    ax.set_xlabel('Meters', fontsize=14)
    savefigp(info="[INFO]: KDE", fname=filename, fig=fig)

def _gather_joint_samples(
    data: Union[Dict[int, pd.DataFrame], pd.DataFrame],
    columns: Sequence[str],
    warn_label: str,
) -> Optional[pd.DataFrame]:
    """
    Normalize inputs (dict or DataFrame) into a single DataFrame containing
    the required columns. Returns None if no usable samples exist.
    """
    columns = list(columns)
    if isinstance(data, dict):
        frames = []
        for nterms, df in data.items():
            if set(columns).issubset(df.columns):
                frames.append(df[columns].assign(nterms=nterms))
        if not frames:
            warnings.warn(f"No realizations supplied for {warn_label}.")
            return None
        df = pd.concat(frames, ignore_index=True)
    else:
        df = data

    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Expected columns {missing} in input data.")

    samples = df[columns].dropna()
    if len(samples) < 5:
        warnings.warn(f"Not enough samples to plot joint KDE for {warn_label}.")
        return None

    return samples


def plot_joint_kde(
    data: Union[Dict[int, pd.DataFrame], pd.DataFrame],
    x_col: str,
    y_col: str,
    filename: str,
    color: str = "r",
    levels: int = 6,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    rug_height: float = -0.15,
) -> None:
    """
    Plot a seaborn-based joint KDE with marginal rug plots for any two columns.
    """
    samples = _gather_joint_samples(data, [x_col, y_col], f"{x_col}/{y_col}")
    if samples is None:
        return

    xlabel = xlabel or x_col
    ylabel = ylabel or y_col
    title = title or f"Joint KDE: {y_col} vs. {x_col}"

    g = sns.jointplot(data=samples, x=x_col, y=y_col, color=color, space=0, alpha=0.3)
    x_vals = samples[x_col].to_numpy()
    y_vals = samples[y_col].to_numpy()
    if len(x_vals) > 10:
        grid_size = 200j
        xx, yy = np.mgrid[
            x_vals.min():x_vals.max():grid_size,
            y_vals.min():y_vals.max():grid_size,
        ]
        kernel = gaussian_kde(np.vstack([x_vals, y_vals]))
        zz = np.reshape(kernel(np.vstack([xx.ravel(), yy.ravel()])), xx.shape)
        g.ax_joint.contourf(
            xx,
            yy,
            zz,
            levels=30,
            cmap="Reds",
            alpha=0.35,
        )
        g.ax_joint.contour(
            xx,
            yy,
            zz,
            levels=6,
            colors="k",
            linewidths=0.5,
            alpha=0.5,
        )
    g.plot_joint(sns.kdeplot, color=color, zorder=0, levels=levels, fill=True, alpha=0.6)
    g.plot_marginals(sns.rugplot, color=color, height=rug_height, clip_on=False)
    g.ax_marg_x.margins(y=0.15)
    g.ax_marg_y.margins(x=0.15)
    ymax = g.ax_marg_x.get_ylim()[1]
    xmax = g.ax_marg_y.get_xlim()[1]
    g.ax_marg_x.set_ylim(0, ymax * 1.05)
    g.ax_marg_y.set_xlim(0, xmax * 1.05)
    g.ax_joint.set_xlabel(xlabel)
    g.ax_joint.set_ylabel(ylabel)

    g.ax_joint.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    g.ax_joint.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    g.ax_marg_x.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    g.ax_marg_y.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    g.figure.suptitle(title)
    g.figure.tight_layout()
    g.figure.subplots_adjust(top=0.92)

    savefigp(fname=filename, fig=g.figure)



# ---------------------------------------------------------------------------- 
#
#                      M  A  I  N        P  R  O  G  R  A  M 
#
# ---------------------------------------------------------------------------- 

if __name__ == "__main__":

    # check clawpack library
    try:
        CLAW = os.environ['CLAW']
    except:
        raise Exception("*** Must first set CLAW enviornment variable")
    
 
    # =======================================================================
    # 1. Setup topography and fault
    # =======================================================================

    # scratch directory for storing topo and dtopo files:
    topo_filename = os.path.join(TOPO_DIR, TOPO_FILENAME)
    shore_filename = os.path.join(TOPO_DIR, SHORE_FILENAME)
    fault_filename = os.path.join(TOPO_DIR, FAULT_FILENAME)
    selected_fault_name = os.path.join(TOPO_DIR, FAULT_FILENAME_SELECTED)

    # 1.1. Read and display topography 
    topo = setup_topo(topo_filename, topo_type=2, make_plot=True)
    shore = np.load(shore_filename)

    # 1.2. Compute geometry for full fault    
    df_fault = pd.read_csv(fault_filename)
    strike_length, dip_length = compute_fault_geometry(df_fault)

    # 1.3. select subfault and compute its geometry   
    df_selected, strike_length_selected, dip_length_selected = select_fault(
            df_fault, dip_range=DIP_RANGE, strike_range=STRIKE_RANGE,
            output_path=selected_fault_name)

    # convert strike and dip length from km to m
    strike_length_selected = 1e3 * strike_length_selected
    dip_length_selected = 1e3 * dip_length_selected

    # 1.4. setup original fault 
    fault_org = setup_fault(fault_filename)

    # 1.4.1. setup selected fault
    fault_selected = setup_fault(selected_fault_name)
    
    # 1.5. Subdivide the fault into smaller subfaults
    fault_subdivided = subdivide_fault(fault_selected, nstrike=NSTRIKE, ndip=NDIP)
    
    plot_subfault(fault_org, fault_selected, fault_subdivided, shore, 
                  filename=SUBFAULT_DIVIDED_FILENAME)

    # =======================================================================
    # 2. Karhunen–Loève (KL) expansion
    # =======================================================================

    # 2.1. Fix desired M0
    M0_desired = fix_M0_desired(Mw_desired=MW_DESIRED)
    print(f"[INFO  2.1]: Desired M0: {M0_desired:.3e} for Mw = {MW_DESIRED}")

    # 2.2. Compute subfault distances
    print(f"[INFO  2.2]: Computing subfault distances\n")   
    D, Dstrike, Ddip = compute_subfault_distances(fault_subdivided)

    # 2.3. correlation matrix: Gaussian with correlation lengths of strike and dip:
    print("[INFO  2.3]: Compute correlation matrix")
    C = compute_correlation_matrix(Dstrike, Ddip, strike_length_selected, dip_length_selected)
    
    # 2.4. compute mean slip from desired Mw
    print("[INFO  2.4]: Compute mean slip")
    mean_slip = compute_mean_slip(fault_subdivided, M0_desired)
    print(f"- Mean slip {mean_slip} meters for Mw {M0_desired}")
    
    # 2.5. Compute eigenvalues and eigenvectors of the covariance matrix
    print("[INFO  2.5]: Compute eigenvalues and eigenvectors of the covariance matrix")
    eigen_value, eigen_vector, mean_slip_g = compute_eigenmodes(C, mean_slip)

    # 2.6. Plot eigen values  
    print("[INFO  2.6]: Plot eigenvalues of the covariance matrix")
    plot_eigenvalues(eigen_value, kplot=20,  filename='eigenvalues-2.png')

    # 2.7. Plot eigen modes
    print("[INFO  2.7]: Plot eigenmodes of the covariance matrix")
    plot_eigenmodes(fault_subdivided, eigen_vector, modes=MODES, 
                    slip_factor=18., filename='CSZmodes-2.png')

    # 2.8. Taper functions
    print("[INFO  2.8]: Apply taper function")
    # au = get_taper_function(taper_name='exp_depth', x, depth0=10e3, MAX_DEPTH = 32500., dip=13., W=100e3, make_plot=None)
    tau = lambda d: 1. - np.exp((d - MAX_DEPTH)*20/MAX_DEPTH)

    
    # 2.9. Compute deformation at area of interest (AOI)
    print("[INFO  2.9]: compute deformation at AOI")
    dZ, dtopo, x_dtopo, y_dtopo = compute_deformation_AOI(fault_subdivided, AOI)

    # Ganti dengan daerah di sulawesi utara
    print("\ninterpolate dtopo around Manado City location:")
    interpolate_point_of_interest(dtopo=dtopo)

     # 2.10.test realization
    print("[INFO 2.10]: Test realizations")
    realizations = test_realizations(NTRIAL_TEST, NTERM_TEST, dZ, y_dtopo, 
                                     fault_subdivided, eigen_value, eigen_vector, 
                                     tau, M0_desired)
    
    # 2.11. Plot realizations
    plot_realizations(fault_subdivided, dtopo, dZ, y_dtopo, eigen_value, eigen_vector, 
                      tau, NTERM_TEST, M0_desired, filename=REALIZATION_FILENAME)
    
    # 2.12. Generate realizations
    print("[INFO 2.11]: Generate realizations")
    random.seed(12345)
    realizations_dict = {}
    for nterms in NTERM_GEN_LIST:
        realizations = test_realizations(NTRIAL_GEN, nterms, dZ, y_dtopo, 
                                         fault_subdivided, eigen_value, eigen_vector, 
                                         tau, M0_desired)
        realizations_dict[nterms] = realizations

    # 2.13. save realization
    print("[INFO 2.13]: Save realizations")
    with open(REALIZATION_PKL_FILENAME, "wb") as f:
        pickle.dump(realizations_dict, f)

    # 2.14. Plot hazard curves
    print("[INFO 2.14]: Plot hazard curve")
    zetai = np.linspace(0, 8, 121)
    plot_hazard_curves(realizations_dict, zetai, filename=HAZARD_CURVE_FILENAME)
    
    # 2.15. Plot KDE at dbshore
    print("[INFO 2.15]: Plot DE at dbshore")
    dzs = np.linspace(-8, 2, 1001)
    plot_kde(realizations_dict, dzs, filename=DB_SHORE_CURVE_FILENAME)
    
    # 2.16. Plot joint KDEs of η_max
    print("[INFO 2.16]: Plot Joint KDE: energy vs. η_max")
    plot_joint_kde(realizations_dict,
        x_col="Energy", y_col="EtaMax",
        filename=KDE_ETAMAX_FILENAME,
        xlabel="||a||^2 (energy)",
        ylabel="Eta Max (m)",
        title="Joint KDE: energy vs. η_max",
    )

    # 2.17. Plot Joint KDE: energy vs. uplift
    print("[INFO 2.17]: Plot Joint KDE: energy vs. uplift")
    plot_joint_kde(realizations_dict,
        x_col="Energy", y_col="uplift",
        filename=KDE_UPLIFT_FILENAME,
        xlabel="||a||^2 (energy)",
        ylabel="uplift (m)",
        title="Joint KDE: energy vs. uplift",
    )

    # ------------------------------------------------------------------
    # sys.exit("\nEXIT PROGRAM AT PLOT EIGENMODES\n")
    # ------------------------------------------------------------------
