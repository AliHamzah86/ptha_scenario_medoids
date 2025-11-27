import pandas as pd
import numpy as np
import os

# ======================================================
# 1. FAULT GEOMETRY COMPUTATION
# ======================================================

def compute_fault_geometry(df: pd.DataFrame):
    """
    Compute:
      - strike_length : total length along strike direction (top depth row)
      - dip_length    : mean dip length across columns
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

    return strike_length, dip_length, total_area


# ======================================================
# 2. FAULT SUBSELECTION + GEOMETRY
# ======================================================

def select_fault(df: pd.DataFrame,
                 dip_range: tuple,
                 strike_range: tuple,
                 output_path: str = "selected_fault.csv"):
    """
    Select a subset of the fault model based on dip and strike indices,
    save result to CSV with the same format as the input,
    and compute geometry for the selected subset.

    Returns:
        df_selected, strike_length, dip_length, total_area
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
    strike_length, dip_length, total_area = compute_fault_geometry(df_selected)

    return df_selected, strike_length, dip_length, total_area


# ======================================================
# 3. USAGE EXAMPLE (MAIN)
# ======================================================
if __name__ == "__main__":

    full_fault_path = "./topo/sulawesi.csv"
    selected_fault_path = "./topo/selected_fault.csv"

    # Load full fault model
    df_fault = pd.read_csv(full_fault_path)

    # Compute geometry for full fault
    sL, dL, A = compute_fault_geometry(df_fault)
    print("Full fault geometry:")
    print(f"  Strike length : {sL:.3f} km")
    print(f"  Dip length    : {dL:.3f} km")
    print(f"  Total area    : {A:.3f} km²")

    # Define dip/strike index ranges for selection
    dip_range = (0, 4)       # example: top 5 dip rows
    strike_range = (10, 30)  # example: columns 10–30

    # If selected file already exists, DO NOT recompute
    if os.path.exists(selected_fault_path):
        print(f"WARNING: File '{selected_fault_path}' already exists. "
              f"Skipping selection and loading existing file.")

        df_selected = pd.read_csv(selected_fault_path)
        sL_sel, dL_sel, A_sel = compute_fault_geometry(df_selected)
    else:
        # Perform selection and save
        df_selected, sL_sel, dL_sel, A_sel = select_fault(
            df_fault,
            dip_range=dip_range,
            strike_range=strike_range,
            output_path=selected_fault_path
        )

    print("\nSelected fault geometry:")
    print(f"  Strike length : {sL_sel:.3f} km")
    print(f"  Dip length    : {dL_sel:.3f} km")
    print(f"  Total area    : {A_sel:.3f} km²")
