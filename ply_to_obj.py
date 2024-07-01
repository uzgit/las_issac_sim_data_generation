#!/usr/bin/env python3

import argparse
import cv2
import numpy
import os
import pymeshlab
import glob

argument_parser = argparse.ArgumentParser(description='Process input file and resolution parameters')
argument_parser.add_argument("input_files", metavar='FILE', type=str, nargs='+', help="Absolute path to the input point cloud file.")
argument_parser.add_argument('--resolution_x', type=int, default=16384, help='Horizontal resolution (default: 16384')
argument_parser.add_argument('--resolution_y', type=int, default=16384, help='Vertical resolution (default: 16384)')
argument_parser.add_argument('--output_file', type=str, default=None, help='Path to the output file (default: None)')

arguments = argument_parser.parse_args()

def process_file( input_file_abs, resolution_x, resolution_y ):

    base, ext = os.path.splitext(input_file_abs)

    output_file = f"{base}.obj"
    texture_file = f"{os.path.basename(base)}.png"

    print(f"Processing {input_file_abs}\n\toutput mesh: {output_file}\n\toutput texture: {texture_file}")
    print(f"\ttexture resolution: {resolution_x}x{resolution_y}")

    meshset = pymeshlab.MeshSet()

    print(f"loading {input_file_abs}")
    mesh = meshset.load_new_mesh(input_file_abs)

    print("computing UV parametrization...")
    meshset.apply_filter("compute_texcoord_parametrization_flat_plane_per_wedge", projectionplane="XY")

    print("mapping texture from vertices to faces...")
    meshset.apply_filter("compute_texmap_from_color", textname=texture_file, textw=resolution_x, texth=resolution_y)

    print("centering mesh on the origin...")
    meshset.apply_filter("compute_matrix_from_translation", traslmethod="Center on Scene BBox")
    print("rotating mesh -90 degrees in X...")
    meshset.apply_filter("compute_matrix_from_rotation", rotaxis="X axis", angle=-90)

    mesh = meshset.current_mesh()
    bounding_box = mesh.bounding_box()
    translate_up_by = -bounding_box.min()[1]

    print("translating mesh upwards so it is in positive z space...")
    meshset.apply_filter("compute_matrix_from_translation", traslmethod="XYZ translation", axisy=translate_up_by)

    print(f"saving mesh to {output_file}")
    meshset.save_current_mesh(output_file)

    print("done!")

# print( arguments.input_files )
# input_files = glob.glob( arguments.input_files )
for pattern in arguments.input_files:
    input_files = glob.glob( pattern )
    for input_file in input_files:
        input_file_abs = os.path.abspath( input_file )
        # print( input_file_abs )
        process_file( input_file_abs, arguments.resolution_x, arguments.resolution_y )
        print()