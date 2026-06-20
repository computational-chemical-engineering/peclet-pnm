"""Verify the SDF segmentation / geometry import."""
import sys
import os
import struct
import numpy as np
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../cfd_utils')))
import pnm
from vti import save_vti

def verify_segmentation(input_file, output_file, edge_file):
    print(f"Reading {input_file}...")
    # New Binding returns: (numpy_array_3d, origin_zyx, spacing_zyx)
    sdf_3d, origin, spacing = pnm.SDFReader.read_vti(input_file)
    
    # Resolution/Shape is now inherent in the array
    shape = sdf_3d.shape
    print(f"Grid Shape (Nz, Ny, Nx): {shape}")
    
    print("Running Segmentation...")
    # Updated binding accepts the 3D array and ZYX spacing
    segmentation_flat = pnm.segment_volume(sdf_3d, spacing)
    
    # Reshape the flat result to our 3D convention
    seg_3d = np.array(segmentation_flat, dtype=np.int32).reshape(shape)
    
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
    # Note: If extract_topology_gpu still needs the {nx, ny, nz} list for its 
    # internal logic, we reverse our ZYX shape back to XYZ: shape[::-1]
    connections = pnm.extract_topology_gpu(segmentation_flat, shape[::-1])
    print(f"Found {len(connections)} connections.")
    
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