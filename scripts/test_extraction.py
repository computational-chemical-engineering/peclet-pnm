"""Smoke test for pore-network extraction from an SDF."""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
import pnm_backend

def test_extraction(filename):
    print(f"Reading {filename}...")
    sdf_data = pnm_backend.SDFReader.read_vti(filename)
    print(f"Loaded SDF. Shape: {sdf_data.resolution}")
    
    print("Extracting pores on GPU...")
    pores = pnm_backend.extract_pores(sdf_data)
    
    print(f"Found {len(pores)} pores.")
    if len(pores) > 0:
        print("First 5 pores:")
        for i in range(min(5, len(pores))):
            print(pores[i])
            
    # Simple validation: Check if duplicate locations?
    import numpy as np
    coords = np.array([[p.x, p.y, p.z] for p in pores])
    if len(coords) > 0:
        print(f"Bounds: Min {coords.min(axis=0)}, Max {coords.max(axis=0)}")
        
    return pores

def save_vtp(filename, pores):
    import struct
    with open(filename, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <PolyData>\n')
        f.write(f'    <Piece NumberOfPoints="{len(pores)}" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">\n')
        
        f.write('      <Points>\n')
        f.write('        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
        for p in pores:
            f.write(f'{p.x} {p.y} {p.z}\n')
        f.write('        </DataArray>\n')
        f.write('      </Points>\n')
        
        f.write('      <PointData Scalars="Radius">\n')
        f.write('        <DataArray type="Float32" Name="Radius" format="ascii">\n')
        for p in pores:
            f.write(f'{p.radius}\n')
        f.write('        </DataArray>\n')
        f.write('      </PointData>\n')
        
        f.write('    </Piece>\n')
        f.write('  </PolyData>\n')
        f.write('</VTKFile>\n')
    print(f"Saved pores to {filename}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pores = test_extraction(sys.argv[1])
        outfile = "pores.vtp"
        save_vtp(outfile, pores)
    else:
        # Default to a file in data/ if exists
        default_file = "data/packing_ring.vti"
        if os.path.exists(default_file):
            pores = test_extraction(default_file)
            save_vtp("pores.vtp", pores)
        else:
            print(f"Usage: {sys.argv[0]} <vti_file>")
