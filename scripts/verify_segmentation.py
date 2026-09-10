"""Verify the SDF segmentation / geometry import.

Run:  PYTHONPATH=<pnm/build> python scripts/verify_segmentation.py <sdf.vti>
Writes the labelled volume (segmentation.vti) and the pore-pair edge list (network.edges).
"""
import argparse
import os

import numpy as np

import peclet.pnm as pnm
from vti import save_vti  # scripts/vti.py — the (Nz,Ny,Nx) VTI writer beside this script

def verify_segmentation(input_file, output_file, edge_file):
    print(f"Reading {input_file}...")
    # New Binding returns: (numpy_array_3d, origin_zyx, spacing_zyx)
    sdf_3d, origin, spacing = pnm.SDFReader.read_vti(input_file)
    
    # Resolution/Shape is now inherent in the array
    shape = sdf_3d.shape
    print(f"Grid Shape (Nz, Ny, Nx): {shape}")
    
    print("Running Segmentation...")
    seg_3d = pnm.segment_volume(sdf_3d, spacing)  # int32 (Nz,Ny,Nx), same shape as the SDF
    
    # Stats logic
    unique_labels = np.unique(seg_3d)
    pores = unique_labels[unique_labels > 0]
    solids = unique_labels[unique_labels < 0]
    
    print(f"Total Labels: {len(unique_labels)}")
    print(f"Pore IDs: {len(pores)}")
    print(f"Solid IDs: {len(solids)}")
    
    # Save output using the general saver
    print(f"Saving segmented volume to {output_file}...")
    save_vti(output_file, 
             fields={"Labels": seg_3d, "SDF": sdf_3d}, 
             spacing=spacing, 
             origin=origin)
    
    print("Extracting Topology...")
    connections = pnm.extract_topology(seg_3d)  # (M,2) int32
    print(f"Found {len(connections)} connections "
          f"({int(((connections[:, 0] > 0) & (connections[:, 1] > 0)).sum())} pore-pore).")
    
    with open(edge_file, "w") as f:
        for u, v in connections:
            f.write(f"{u} {v}\n")
    print(f"Saved {edge_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PNM Segmentation Verification")
    parser.add_argument("input", help="Input SDF VTI file")
    parser.add_argument("-o", "--output", default="segmentation.vti", help="Output Labels VTI")
    parser.add_argument("-e", "--edges", default="network.edges", help="Output edge list")
    
    args = parser.parse_args()

    if os.path.exists(args.input):
        verify_segmentation(args.input, args.output, args.edges)
    else:
        print(f"Error: {args.input} not found.")