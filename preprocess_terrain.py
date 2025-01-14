#!/usr/bin/env python3

import argparse
import multiprocessing
import numpy
import os
import subprocess
import glob
import shutil
import sys

import cloudComPy as cc
import cloudComPy.PoissonRecon
from gendata import getSampleCloud2, getSamplePoly, dataDir

# Initialize CloudCompare
cc.initCC()

#print(f"{sys.argv[1]}")

parser = argparse.ArgumentParser(description="Perform Poisson Surface Reconstruction on a point cloud")
parser.add_argument("input_files", metavar='FILE', type=str, nargs='+', help="Absolute path to the input point cloud file.")
#parser.add_argument("--output_file", default=None, help="Absolute path to the output .ply file")
parser.add_argument("--filter_point_density", type=float, default=8, help="Minimum point density when filtering Poisson-reconstructed meshes.")
parser.add_argument("--mesh_sampling_density", type=float, default=1000, help="Point density when sampling meshes.")
parser.add_argument("--subsample_percentage", type=int, default=None, help="Percentage to subsample the point cloud.")
args = parser.parse_args()

def process_file( input_file_abs ):
    # Generate output file name
    base, ext = os.path.splitext(input_file_abs)
    
    rgb_output_file_abs   = base + "_rgb.ply"
    rgb_filtered_output_file_abs   = base + "_rgb_filtered.ply"
    label_output_file_abs = base + "_label.ply"
    label_filtered_output_file_abs = base + "_label_filtered.ply"

    if( os.path.exists(rgb_output_file_abs) and os.path.exists(label_output_file_abs) ):
        print(f"already processed {input_file_abs}")
        return

    print(f"{input_file_abs} --> {rgb_output_file_abs}")

    print(f"loading input file {input_file_abs}...")
    cloud = cc.loadPointCloud(input_file_abs)

    print(f"{cloud.size()=}")
    if( args.subsample_percentage is not None ):
        assert 0 < args.subsample_percentage < 100

        num_points = int(cloud.size() * args.subsample_percentage / 100)
        print(f"subsampling randomly to {args.subsample_percentage}% = {num_points}")
        cloud = cc.CloudSamplingTools.subsampleCloudRandomly(cloud, num_points)
        print(f"{cloud.size()=}")

    print(f"shifting cloud to center around (x=0, y=0) and min(z) = 0")
    boundingBox = cloud.getOwnBB()
    cmin = numpy.array(boundingBox.minCorner())
    cmax = numpy.array(boundingBox.maxCorner())
    center = 0.5 * (cmin + cmax)
    translation = (-center[0], -center[1], -cmin[0]) 
    cloud.translate(translation)

    print("computing normals...")
    cc.computeNormals([cloud], defaultRadius=0.1, model=cc.LOCAL_MODEL_TYPES.QUADRIC, orientNormals=True, preferredOrientation=cc.Orientation.PLUS_Z)
    
    for name, index in cloud.getScalarFieldDic().items():
        cloud.renameScalarField( index, "".join(c for c in name.lower() if 'a' <= c <= 'z') )
    orig_scalar_field_indices = cloud.getScalarFieldDic()
    #print(f"{orig_scalar_field_indices=}")


    #print("orienting normals...")
    #cloud.orientNormalsWithMST(octreeLevel=10)
    #cloud.orientNormalsWithFM(octreeLevel=11)

    print("doing Poisson reconstruction...")
    mesh = cc.PoissonRecon.PR.PoissonReconstruction(pc=cloud, threads=multiprocessing.cpu_count(), density=True, withColors=True, depth=10, samplesPerNode=2)

    if( "classification" in orig_scalar_field_indices ):

        parameters = cc.interpolatorParameters()
        parameters.method = cc.INTERPOL_METHOD.RADIUS
        parameters.algos = cc.INTERPOL_ALGO.NORMAL_DIST
        parameters.radius = 0.1
        parameters.sigma = 0.04
        print(f"found manual classification scalar field in input cloud ({orig_scalar_field_indices=})")
        ret = cc.interpolateScalarFieldsFrom(mesh.getAssociatedCloud(), cloud, list(range(len(cloud.getScalarFieldDic()))), parameters)
        print(f"interpolated manual classification onto the sampled cloud!\n\t{ret=}")

    ret = cc.SaveMesh(mesh, rgb_output_file_abs)

    print(f"saving filtered mesh to {rgb_output_file_abs}...")
    command = f"CloudCompare -SILENT -O {rgb_output_file_abs} -AUTO_SAVE OFF -FILTER_SF {args.filter_point_density} MAX -M_EXPORT_FMT PLY -PLY_EXPORT_FMT ASCII -SAVE_MESHES FILE {rgb_output_file_abs}"
    command = command.split(" ")
    result = subprocess.run(command, capture_output=True, text=True)
    print(f"saved filtered mesh!")

    # reload mesh to get the filtered version
    print(f"reloading filtered mesh...")
    mesh = cc.loadMesh(rgb_output_file_abs)

    print(f"sampling mesh with {args.mesh_sampling_density=}")
    sampled_cloud = mesh.samplePoints(densityBased=True, samplingParameter=args.mesh_sampling_density, withNormals=True, withRGB=True, withTexture=True)

    radius = 0.5
    print(f"calculating verticality, {radius=}")
    cc.computeFeature(cc.GeomFeature.Verticality, radius, [sampled_cloud])
    
    radius = 0.01
    print(f"calculating planarity, {radius=}")
    cc.computeFeature(cc.GeomFeature.Planarity, radius, [sampled_cloud])
    #print(f"calculating omnivariance, {radius=}")
    #cc.computeFeature(cc.GeomFeature.Omnivariance, radius, [sampled_cloud])
    
    radius = 2
    print(f"calculating surfacevariation, {radius=}")
    cc.computeFeature(cc.GeomFeature.SurfaceVariation, radius, [sampled_cloud])

    print(f"{sampled_cloud.getScalarFieldDic()=}")

    # rename the scalar fields to get rid of extraneous data like radii and uppercase/lowercase, which are objectively excessive things to add to names like this
    for name, index in sampled_cloud.getScalarFieldDic().items():
        sampled_cloud.renameScalarField( index, "".join(c for c in name.lower() if 'a' <= c <= 'z') )
    scalar_field_indices = sampled_cloud.getScalarFieldDic()
    print(f"{scalar_field_indices=}")

    print("interpolating all scalar fields from the pointcloud to the mesh...")
    parameters = cc.interpolatorParameters()
    parameters.method = cc.INTERPOL_METHOD.RADIUS
    parameters.algos = cc.INTERPOL_ALGO.NORMAL_DIST
    parameters.radius = 0.1
    parameters.sigma = 0.04
    ret = cc.interpolateScalarFieldsFrom(mesh.getAssociatedCloud(), sampled_cloud, list(range(len(sampled_cloud.getScalarFieldDic()))), parameters)
    print(f"{ret=}")

    scalar_field_indices = mesh.getAssociatedCloud().getScalarFieldDic()
    print(f"{scalar_field_indices=}")

    try:
        # get planarity
        planarity = mesh.getAssociatedCloud().getScalarField(scalar_field_indices["planarity"]).toNpArray()
        planarity = numpy.nan_to_num(planarity, nan=0)
        print(f"{planarity.min()=} : {planarity.max()=}")
        
        # get surface_variation
        surface_variation = mesh.getAssociatedCloud().getScalarField(scalar_field_indices["surfacevariation"]).toNpArray()
        surface_variation = numpy.nan_to_num(surface_variation, nan=0)
        print(f"{surface_variation.min()=} : {surface_variation.max()=}")
    except:
        file_dir = os.path.dirname(input_file_abs)
        target_dir = os.path.join(file_dir, "not_processed")
        target_file_path = os.path.join(target_dir, os.path.basename(input_file_abs))
        shutil.move(input_file_abs, target_file_path)
        print(f"\033[91m planarity error.\nMoving\n'{input_file_abs}'\nto\n'{target_file_path}'\033[0m")
        return

    # get verticality and normalize
    verticality = mesh.getAssociatedCloud().getScalarField(scalar_field_indices["verticality"]).toNpArray()
    verticality = numpy.nan_to_num(verticality, nan=0)
    print(f"{verticality.min()=} : {verticality.max()=}")


    #label = numpy.zeros_like(verticality)
    label = numpy.ones_like(verticality)
    #label[planarity   > 0.95] = 1
    label[verticality > 0.01] = 0
    #label[verticality > 0.00000002] = 0
    label[surface_variation > 0.002] = 0

    if( "classification" in scalar_field_indices.keys() ):
        print(f"found manual classification scalar field! applying to label...")
        classification = mesh.getAssociatedCloud().getScalarField(scalar_field_indices["classification"]).toNpArray()
        label[classification == 1] = 0
    else:
        print(f"no manual classification scalar field found.")
        print(f"{scalar_field_indices=}")

    print(f"{label.min()=} : {label.max()=}, {label.shape=}, {label.sum()=}")
    label_scalar_field_index = mesh.getAssociatedCloud().addScalarField("label")
    label_scalar_field = mesh.getAssociatedCloud().getScalarField( label_scalar_field_index )
    label_scalar_field.fromNpArrayCopy( label )

    
    if( True ):
        processed_cloud = mesh.getAssociatedCloud()

        sigma = 2

        #mesh.getAssociatedCloud().applyScalarFieldGaussianFilter( SFindex=label_scalar_field_index, sigma=2)
        print(f"smoothing label: Gaussian with {sigma=}")
        processed_cloud.applyScalarFieldGaussianFilter( SFindex=label_scalar_field_index, sigma=sigma)
        
        for name, index in processed_cloud.getScalarFieldDic().items():
            processed_cloud.renameScalarField( index, "".join(c for c in name.lower() if 'a' <= c <= 'z') )
        scalar_field_indices = processed_cloud.getScalarFieldDic()
        print(f"{scalar_field_indices=}")
        
        label = mesh.getAssociatedCloud().getScalarField(scalar_field_indices["labelsmooth"]).toNpArray()

    label *= 255
    colors2 = numpy.vstack([label, label, label, numpy.full(label.shape, 255)]).astype(numpy.uint8).T
    mesh.getAssociatedCloud().colorsFromNPArray_copy(colors2)

    print(f"saving mesh to {label_output_file_abs}")
    ret = cc.SaveMesh(mesh, label_output_file_abs)
    if ret:
        print(f"Mesh saved successfully to {label_output_file_abs}")
    else:
        print(f"Failed to save mesh to {label_output_file_abs}")

# Convert input file path to absolute path
#input_files_abs = os.path.abspath(args.input_file)
#input_files = glob.glob( args.input_files )

for input_file in args.input_files:
    
    input_file_abs = os.path.abspath(input_file)
    process_file( input_file_abs )
    print()
