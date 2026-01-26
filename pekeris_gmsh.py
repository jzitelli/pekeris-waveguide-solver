"""
Gmsh mesh generation for Pekeris waveguide.

Generates a 2D mesh representing an axisymmetric cross-section of the Pekeris
waveguide for use with DOLFINx. The geometry consists of:
- Water layer (depth 0 to H)
- Sediment layer (depth H to z_max)
- Semicircular exclusion around the source at (r=0, z=z_s)
- PML (Perfectly Matched Layer) absorbing regions

PML regions:
- PML_WATER_RIGHT: radial absorbing layer in water (r_max to r_max + pml_r)
- PML_SEDIMENT_RIGHT: radial absorbing layer in sediment
- PML_BOTTOM: vertical absorbing layer at bottom (z_max to z_max + pml_z)
- PML_CORNER: corner region where radial and vertical PMLs meet

Coordinate system:
- r (horizontal): radial distance from axis of symmetry
- z (vertical): depth, positive downward from surface
"""

import gmsh
import numpy as np


# Physical tag constants for domains
WATER_DOMAIN = 1
SEDIMENT_DOMAIN = 2
PML_WATER_RIGHT = 3     # PML in water layer (right side)
PML_SEDIMENT_RIGHT = 4  # PML in sediment layer (right side)
PML_BOTTOM = 5          # PML at bottom
PML_CORNER = 6          # PML corner region

# Physical tag constants for boundaries
SOURCE_BOUNDARY = 10
TOP_BOUNDARY = 11       # Pressure-release surface (z=0)
BOTTOM_BOUNDARY = 12    # Truncation boundary (z=z_max + pml_z)
RIGHT_BOUNDARY = 13     # Far-field boundary (r=r_max + pml_r)
INTERFACE = 14          # Water-sediment interface (z=H)
AXIS_BOUNDARY = 15      # Axis of symmetry (r=0)


def gmsh_pekeris_waveguide(
    model: gmsh.model,
    name: str,
    H: float = 150.0,
    z_s: float = 30.0,
    r_max: float = 1000.0,
    z_max: float = 225.0,
    r_source: float = 5.0,
    pml_r: float = 100.0,
    pml_z: float = 100.0,
    lc_fine: float = 2.0,
    lc_coarse: float = 20.0,
) -> gmsh.model:
    """Create a Gmsh model of a Pekeris waveguide cross-section with PML.

    Parameters
    ----------
    model : gmsh.model
        Gmsh model to add the mesh to.
    name : str
        Name (identifier) of the mesh to add.
    H : float
        Water layer depth (m).
    z_s : float
        Source depth (m), must be < H.
    r_max : float
        Maximum radial extent of physical domain (m), before PML.
    z_max : float
        Maximum depth of physical domain (m), must be > H, before PML.
    r_source : float
        Radius of circular exclusion around source (m).
    pml_r : float
        Thickness of radial PML layer (m). Set to 0 to disable.
    pml_z : float
        Thickness of vertical PML layer (m). Set to 0 to disable.
    lc_fine : float
        Characteristic length (mesh size) near the source.
    lc_coarse : float
        Characteristic length (mesh size) at far boundaries.

    Returns
    -------
    gmsh.model
        Gmsh model with the Pekeris waveguide mesh added.
    """
    if z_s >= H:
        raise ValueError(f"Source depth z_s={z_s} must be less than water depth H={H}")
    if z_max <= H:
        raise ValueError(f"Domain depth z_max={z_max} must be greater than water depth H={H}")
    if r_source >= r_max:
        raise ValueError(f"Source radius r_source={r_source} must be less than r_max={r_max}")

    # Total domain extent including PML
    r_total = r_max + pml_r
    z_total = z_max + pml_z

    model.add(name)
    model.setCurrent(name)

    # Use OpenCASCADE kernel for boolean operations
    occ = model.occ

    # Create physical domain rectangles
    # Water layer: (0, 0) to (r_max, H)
    water_rect = occ.addRectangle(0, 0, 0, r_max, H)
    # Sediment layer: (0, H) to (r_max, z_max)
    sediment_rect = occ.addRectangle(0, H, 0, r_max, z_max - H)

    # Create PML rectangles
    pml_rects = []
    if pml_r > 0:
        # Right PML for water: (r_max, 0) to (r_total, H)
        pml_water_right_rect = occ.addRectangle(r_max, 0, 0, pml_r, H)
        pml_rects.append(('pml_water_right', pml_water_right_rect))
        # Right PML for sediment: (r_max, H) to (r_total, z_max)
        pml_sediment_right_rect = occ.addRectangle(r_max, H, 0, pml_r, z_max - H)
        pml_rects.append(('pml_sediment_right', pml_sediment_right_rect))

    if pml_z > 0:
        # Bottom PML: (0, z_max) to (r_max, z_total)
        pml_bottom_rect = occ.addRectangle(0, z_max, 0, r_max, pml_z)
        pml_rects.append(('pml_bottom', pml_bottom_rect))

    if pml_r > 0 and pml_z > 0:
        # Corner PML: (r_max, z_max) to (r_total, z_total)
        pml_corner_rect = occ.addRectangle(r_max, z_max, 0, pml_r, pml_z)
        pml_rects.append(('pml_corner', pml_corner_rect))

    # Create semicircular exclusion around source at (0, z_s)
    source_disk = occ.addDisk(0, z_s, 0, r_source, r_source)

    # Cut the source exclusion from the water layer
    water_cut = occ.cut([(2, water_rect)], [(2, source_disk)])
    occ.synchronize()

    # Get the resulting water domain tag
    water_surfaces = [tag for dim, tag in water_cut[0] if dim == 2]
    if len(water_surfaces) != 1:
        raise RuntimeError(f"Expected 1 water surface after cut, got {len(water_surfaces)}")
    water_tag = water_surfaces[0]

    # Collect all surfaces for fragmentation
    all_surfaces = [(2, water_tag), (2, sediment_rect)]
    for pml_name, pml_tag in pml_rects:
        all_surfaces.append((2, pml_tag))

    # Fragment to create shared interfaces
    fragmented = occ.fragment(all_surfaces, [])
    occ.synchronize()

    # Identify the resulting surfaces by their center of mass
    surfaces = model.getEntities(2)

    water_tags = []
    sediment_tags = []
    pml_water_right_tags = []
    pml_sediment_right_tags = []
    pml_bottom_tags = []
    pml_corner_tags = []

    tol = 1e-6

    for dim, tag in surfaces:
        com_x, com_y, com_z = occ.getCenterOfMass(dim, tag)

        # Classify by position
        in_pml_r = com_x > r_max - tol
        in_pml_z = com_y > z_max - tol
        in_water_depth = com_y < H - tol
        in_sediment_depth = H + tol < com_y < z_max - tol

        if in_pml_r and in_pml_z:
            pml_corner_tags.append(tag)
        elif in_pml_r and in_water_depth:
            pml_water_right_tags.append(tag)
        elif in_pml_r and in_sediment_depth:
            pml_sediment_right_tags.append(tag)
        elif in_pml_z:
            pml_bottom_tags.append(tag)
        elif in_water_depth:
            water_tags.append(tag)
        else:
            sediment_tags.append(tag)

    # Add physical groups for domains
    model.addPhysicalGroup(2, water_tags, tag=WATER_DOMAIN)
    model.setPhysicalName(2, WATER_DOMAIN, "Water")

    model.addPhysicalGroup(2, sediment_tags, tag=SEDIMENT_DOMAIN)
    model.setPhysicalName(2, SEDIMENT_DOMAIN, "Sediment")

    if pml_water_right_tags:
        model.addPhysicalGroup(2, pml_water_right_tags, tag=PML_WATER_RIGHT)
        model.setPhysicalName(2, PML_WATER_RIGHT, "PML Water Right")

    if pml_sediment_right_tags:
        model.addPhysicalGroup(2, pml_sediment_right_tags, tag=PML_SEDIMENT_RIGHT)
        model.setPhysicalName(2, PML_SEDIMENT_RIGHT, "PML Sediment Right")

    if pml_bottom_tags:
        model.addPhysicalGroup(2, pml_bottom_tags, tag=PML_BOTTOM)
        model.setPhysicalName(2, PML_BOTTOM, "PML Bottom")

    if pml_corner_tags:
        model.addPhysicalGroup(2, pml_corner_tags, tag=PML_CORNER)
        model.setPhysicalName(2, PML_CORNER, "PML Corner")

    # Get all boundary curves and classify them
    all_curves = model.getEntities(1)

    source_curves = []
    top_curves = []
    bottom_curves = []
    right_curves = []
    interface_curves = []
    axis_curves = []

    for dim, tag in all_curves:
        # Get bounding box of curve
        xmin, ymin, zmin, xmax, ymax, zmax_bb = occ.getBoundingBox(dim, tag)

        # Check if curve is on the source circle (semicircle at r=0, centered at z_s)
        # Source curves have xmax <= r_source and are curved
        if xmax <= r_source + tol and xmin < tol:
            # Could be source or axis - check if it's curved
            curve_type = model.getType(dim, tag)
            if "Circle" in curve_type or "Ellipse" in curve_type:
                source_curves.append(tag)
                continue

        # Horizontal curves
        if abs(ymax - ymin) < tol:
            y_val = ymin
            if abs(y_val) < tol:  # z = 0 (top surface)
                top_curves.append(tag)
            elif abs(y_val - H) < tol:  # z = H (interface, not in PML region)
                if xmax <= r_max + tol:
                    interface_curves.append(tag)
            elif abs(y_val - z_total) < tol:  # z = z_total (bottom of PML)
                bottom_curves.append(tag)
            continue

        # Vertical curves
        if abs(xmax - xmin) < tol:
            x_val = xmin
            if abs(x_val) < tol:  # r = 0 (axis)
                axis_curves.append(tag)
            elif abs(x_val - r_total) < tol:  # r = r_total (right boundary of PML)
                right_curves.append(tag)

    # Add physical groups for boundaries
    if source_curves:
        model.addPhysicalGroup(1, source_curves, tag=SOURCE_BOUNDARY)
        model.setPhysicalName(1, SOURCE_BOUNDARY, "Source")

    if top_curves:
        model.addPhysicalGroup(1, top_curves, tag=TOP_BOUNDARY)
        model.setPhysicalName(1, TOP_BOUNDARY, "Top (pressure release)")

    if bottom_curves:
        model.addPhysicalGroup(1, bottom_curves, tag=BOTTOM_BOUNDARY)
        model.setPhysicalName(1, BOTTOM_BOUNDARY, "Bottom")

    if right_curves:
        model.addPhysicalGroup(1, right_curves, tag=RIGHT_BOUNDARY)
        model.setPhysicalName(1, RIGHT_BOUNDARY, "Right (far field)")

    if interface_curves:
        model.addPhysicalGroup(1, interface_curves, tag=INTERFACE)
        model.setPhysicalName(1, INTERFACE, "Interface")

    if axis_curves:
        model.addPhysicalGroup(1, axis_curves, tag=AXIS_BOUNDARY)
        model.setPhysicalName(1, AXIS_BOUNDARY, "Axis")

    # Set mesh size fields for refinement near source
    # Field 1: Distance from source curves
    model.mesh.field.add("Distance", 1)
    model.mesh.field.setNumbers(1, "CurvesList", source_curves)
    model.mesh.field.setNumber(1, "Sampling", 100)

    # Field 2: Threshold based on distance
    model.mesh.field.add("Threshold", 2)
    model.mesh.field.setNumber(2, "InField", 1)
    model.mesh.field.setNumber(2, "SizeMin", lc_fine)
    model.mesh.field.setNumber(2, "SizeMax", lc_coarse)
    model.mesh.field.setNumber(2, "DistMin", r_source)
    model.mesh.field.setNumber(2, "DistMax", 10 * r_source)

    # Field 3: Background mesh size
    model.mesh.field.add("Constant", 3)
    model.mesh.field.setNumber(3, "VIn", lc_coarse)

    # Field 4: Minimum of threshold and background
    model.mesh.field.add("Min", 4)
    model.mesh.field.setNumbers(4, "FieldsList", [2, 3])

    model.mesh.field.setAsBackgroundMesh(4)

    # Disable default mesh size from points
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    # Generate 2D mesh
    model.mesh.generate(2)

    return model


def create_mesh_xdmf(model: gmsh.model, name: str, filename: str):
    """Create a DOLFINx mesh from the Gmsh model and write to XDMF.

    Parameters
    ----------
    model : gmsh.model
        Gmsh model containing the mesh.
    name : str
        Name of the mesh in the model.
    filename : str
        Output XDMF filename.
    """
    from mpi4py import MPI
    from dolfinx.io import XDMFFile
    from dolfinx.io import gmsh as gmshio

    model.setCurrent(name)
    mesh_data = gmshio.model_to_mesh(model, MPI.COMM_WORLD, rank=0)
    mesh_data.mesh.name = name

    if mesh_data.cell_tags is not None:
        mesh_data.cell_tags.name = f"{name}_cells"
    if mesh_data.facet_tags is not None:
        mesh_data.facet_tags.name = f"{name}_facets"

    with XDMFFile(mesh_data.mesh.comm, filename, "w") as file:
        file.write_mesh(mesh_data.mesh)
        if mesh_data.cell_tags is not None:
            file.write_meshtags(
                mesh_data.cell_tags,
                mesh_data.mesh.geometry,
                geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{name}']/Geometry",
            )
        if mesh_data.facet_tags is not None:
            file.write_meshtags(
                mesh_data.facet_tags,
                mesh_data.mesh.geometry,
                geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{name}']/Geometry",
            )

    return mesh_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Pekeris waveguide mesh")
    parser.add_argument("--H", type=float, default=150.0, help="Water depth (m)")
    parser.add_argument("--z-s", type=float, default=30.0, help="Source depth (m)")
    parser.add_argument("--r-max", type=float, default=1000.0, help="Max radial extent (m)")
    parser.add_argument("--z-max", type=float, default=225.0, help="Max depth (m)")
    parser.add_argument("--r-source", type=float, default=5.0, help="Source exclusion radius (m)")
    parser.add_argument("--pml-r", type=float, default=100.0, help="Radial PML thickness (m)")
    parser.add_argument("--pml-z", type=float, default=100.0, help="Vertical PML thickness (m)")
    parser.add_argument("--lc-fine", type=float, default=2.0, help="Fine mesh size (m)")
    parser.add_argument("--lc-coarse", type=float, default=20.0, help="Coarse mesh size (m)")
    parser.add_argument("--output", type=str, default="pekeris_mesh", help="Output filename (without extension)")
    parser.add_argument("--view", action="store_true", help="Open Gmsh GUI to view mesh")
    parser.add_argument("--xdmf", action="store_true", help="Also export to XDMF (requires DOLFINx)")
    args = parser.parse_args()

    # Initialize Gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)

    # Create model
    model = gmsh.model()

    print(f"Creating Pekeris waveguide mesh:")
    print(f"  Water depth H = {args.H} m")
    print(f"  Source depth z_s = {args.z_s} m")
    print(f"  Physical domain: r in [0, {args.r_max}] m, z in [0, {args.z_max}] m")
    print(f"  PML thickness: radial={args.pml_r} m, vertical={args.pml_z} m")
    print(f"  Total domain: r in [0, {args.r_max + args.pml_r}] m, z in [0, {args.z_max + args.pml_z}] m")
    print(f"  Source exclusion radius = {args.r_source} m")
    print(f"  Mesh sizes: fine={args.lc_fine} m, coarse={args.lc_coarse} m")

    model = gmsh_pekeris_waveguide(
        model,
        "Pekeris",
        H=args.H,
        z_s=args.z_s,
        r_max=args.r_max,
        z_max=args.z_max,
        r_source=args.r_source,
        pml_r=args.pml_r,
        pml_z=args.pml_z,
        lc_fine=args.lc_fine,
        lc_coarse=args.lc_coarse,
    )

    # Write mesh file
    msh_filename = f"{args.output}.msh"
    gmsh.write(msh_filename)
    print(f"Wrote mesh to {msh_filename}")

    # Export to XDMF if requested
    if args.xdmf:
        xdmf_filename = f"{args.output}.xdmf"
        print(f"Exporting to XDMF: {xdmf_filename}")
        create_mesh_xdmf(model, "Pekeris", xdmf_filename)
        print(f"Wrote XDMF to {xdmf_filename}")

    # Open GUI if requested
    if args.view:
        gmsh.fltk.run()

    gmsh.finalize()
