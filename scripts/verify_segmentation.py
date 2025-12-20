import sys
import os
import struct
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
import pnm_backend

def save_vti_labeled(filename, labels, shape, origin, spacing):
    # Save int32 labeled volume
    nx, ny, nz = shape
    data_bytes = labels.tobytes()
    
    header = f"""<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt64">
  <ImageData WholeExtent="0 {nx-1} 0 {ny-1} 0 {nz-1}" Origin="{origin[0]} {origin[1]} {origin[2]}" Spacing="{spacing[0]} {spacing[1]} {spacing[2]}">
    <Piece Extent="0 {nx-1} 0 {ny-1} 0 {nz-1}">
      <PointData Scalars="Labels">
        <DataArray type="Int32" Name="Labels" format="appended" offset="0"/>
      </PointData>
      <CellData>
      </CellData>
    </Piece>
  </ImageData>
  <AppendedData encoding="raw">
_"""
    with open(filename, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(struct.pack('<Q', len(data_bytes)))
        f.write(data_bytes)
    print(f"Saved labeled volume to {filename}")

def verify_segmentation(filename):
    print(f"Reading {filename}...")
    sdf_data = pnm_backend.SDFReader.read_vti(filename)
    
    print("Running Segmentation...")
    segmentation = pnm_backend.segment_volume(sdf_data)
    
    seg_array = np.array(segmentation, dtype=np.int32)
    # Convert flattened to 3D
    # Note: Pybind returned flat vector.
    # Dimensions: Z, Y, X
    nz, ny, nx = sdf_data.resolution[2], sdf_data.resolution[1], sdf_data.resolution[0]
    
    print(f"Segmentation Elements: {len(seg_array)}")
    
    pores = np.unique(seg_array[seg_array > 0])
    solids = np.unique(seg_array[seg_array < 0])
    
    print(f"Total Labels: {len(np.unique(seg_array))}")
    print(f"Pore IDs: {len(pores)}")
    print(f"Solid IDs: {len(solids)}")
    
    # Save output
    save_vti_labeled("segmentation.vti", seg_array, sdf_data.resolution, sdf_data.origin, sdf_data.spacing)
    
    print("Extracting Topology...")
    connections = pnm_backend.extract_topology(segmentation, sdf_data.resolution)
    print(f"Found {len(connections)} connections.")
    
    # Save topology? simple text file or edge list
    with open("network.edges", "w") as f:
        for u, v in connections:
            f.write(f"{u} {v}\n")
    print("Saved network.edges")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        verify_segmentation(sys.argv[1])
    else:
        # Default to a file in data/ if exists
        default_file = "data/packing_ring.vti"
        if os.path.exists(default_file):
            verify_segmentation(default_file)
        else:
            print(f"Usage: {sys.argv[0]} <vti_file>")
