"""
VTI (VTK ImageData) file I/O utilities.

Provides functions to save field data in VTK ImageData format for visualization
with ParaView and other VTK-compatible tools.
"""

import numpy as np

def save_vti(filename, fields, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    """
    General VTI saver that infers grid dimensions from the input fields.
    
    Parameters
    ----------
    fields : dict { "Name": ndarray }
        Arrays should be shape (Nz, Ny, Nx) for scalars 
        or (Nz, Ny, Nx, 3) for vectors.
    spacing : tuple (dz, dy, dx)
    origin : tuple (oz, oy, ox)
    """
    # 1. Infer shape from the first field
    first_field = next(iter(fields.values()))
    shape_3d = first_field.shape[:3] # Get (Nz, Ny, Nx)
    nz, ny, nx = shape_3d
    
    # 2. Map to VTK Point-Extents (0 to N points for N cells)
    # VTK XML order is ALWAYS X, Y, Z
    ex, ey, ez = nx, ny, nz
    dz, dy, dx = spacing
    oz, oy, ox = origin

    with open(filename, 'wb') as f:
        # XML Boilerplate
        f.write(b'<?xml version="1.0"?>\n')
        f.write(b'<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n')
        f.write(f'  <ImageData WholeExtent="0 {ex} 0 {ey} 0 {ez}" Origin="{ox} {oy} {oz}" Spacing="{dx} {dy} {dz}">\n'.encode('ascii'))
        f.write(f'    <Piece Extent="0 {ex} 0 {ey} 0 {ez}">\n'.encode('ascii'))
        f.write(b'      <CellData>\n')

        # 3. Write Metadata and calculate offsets
        offset = 0
        data_to_write = []
        
        for name, arr in fields.items():
            if arr.shape[:3] != shape_3d:
                raise ValueError(f"Field '{name}' shape {arr.shape} inconsistent with {shape_3d}")
            
            # Determine Types
            if np.issubdtype(arr.dtype, np.integer):
                dtype_str = "Int32"
                arr_fixed = arr.astype('<i4')
            elif arr.dtype == np.float64:
                dtype_str = "Float64"
                arr_fixed = arr.astype('<f8')
            else:
                dtype_str = "Float32"
                arr_fixed = arr.astype('<f4')
            
            n_comp = 3 if arr.ndim == 4 else 1
            n_bytes = arr_fixed.nbytes
            
            f.write(f'        <DataArray type="{dtype_str}" Name="{name}" NumberOfComponents="{n_comp}" format="appended" offset="{offset}"/>\n'.encode('ascii'))
            
            data_to_write.append(arr_fixed)
            offset += n_bytes + 8 # Data size + 8-byte length header

        f.write(b'      </CellData>\n    </Piece>\n  </ImageData>\n')
        
        # 4. Appended Binary Section
        f.write(b'  <AppendedData encoding="raw">\n_')
        
        for arr in data_to_write:
            # Because arrays are (Nz, Ny, Nx), ravel() is naturally X-fastest.
            # For vectors (Nz, Ny, Nx, 3), ravel() produces [u0,v0,w0, u1,v1,w1...]
            flat_bytes = arr.tobytes()
            f.write(np.array([len(flat_bytes)], dtype='<u8').tobytes())
            f.write(flat_bytes)

        f.write(b'\n  </AppendedData>\n</VTKFile>\n')