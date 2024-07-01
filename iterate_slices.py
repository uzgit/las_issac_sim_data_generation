#!/usr/bin/env python3

import asyncio
import argparse
import glob
import numpy
from PIL import Image
import sys
import omni
from omni.isaac.kit import SimulationApp
import time
from datetime import datetime

config = {
    "launch_config": {
        "renderer": "RayTracedLighting",
        "headless": False,
    },
    "resolution": [512, 512],
    "rt_subframes": 2,
    "num_frames": 20,
    "scope_name": "/MyScope",
    "writer": "BasicWriter",
    "writer_config": {
        "output_dir": "_out_offline_generation",
        "rgb": True,
        "bounding_box_2d_tight": False,
        "semantic_segmentation": False,
        "distance_to_image_plane": False,
        "bounding_box_3d": False,
        "occlusion": False,
    },
}

kit = SimulationApp(launch_config=config["launch_config"])
omni.kit.app.get_app().print_and_log("Hello World!")

from omni.isaac.core import World
from pxr import UsdGeom, Gf
from pxr import Usd, UsdGeom
from pxr import UsdShade, Sdf
from PIL import Image
import omni.isaac.core.utils.prims as prims_utils
import omni.replicator.core as rep
from omni.replicator.core import AnnotatorRegistry, BackendDispatch, Writer, WriterRegistry
from omni.isaac.core.utils.viewports import set_camera_view
import omni.isaac.core.utils.prims as prim_utils
import carb

from omni.isaac.core.utils import prims
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import get_current_stage, open_stage
import omni.isaac.core.utils.bounds as bounds_utils
from pxr import Gf

import cv2
import asyncio

parser = argparse.ArgumentParser(description="Process the world_directory path.")
parser.add_argument('world_directory', type=str, help='The path to the world directory')
args = parser.parse_args()

def init_world():

    global world
    world = World(stage_units_in_meters=1.0)

    light_1 = prim_utils.create_prim(
        "/World/Light_1",
        "SphereLight",
        position=numpy.array([0.0, 0.0, 1000.0]),
        attributes={
            "inputs:radius": 1,
            "inputs:intensity": 5e8,
            "inputs:color": (1.0, 1.0, 1.0)
        }
    )

    # disable a lot of the viewport navigational display things
    opts = carb.settings.get_settings().set("persistent/app/viewport/displayOptions", 0)

def clear_world():

    global world
    world.clear()

def load_mesh( prim_path, usd_path ):

    global terrain_prim_path, label_prim_path

    stage = omni.usd.get_context().get_stage()
    prim = UsdGeom.Mesh.Define(stage, prim_path )
    prim.GetPrim().GetReferences().AddReference( usd_path )
    xform = UsdGeom.Xform(prim)
    rotation_matrix = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 90))
    xformable = UsdGeom.Xformable(prim)
    xformable.AddTransformOp().Set(rotation_matrix)

    return prim

def load_world(world_index):
    
    rgb_usds = sorted(list(glob.glob(f"{args.world_directory}/*rgb*.usd")))
    label_usds = sorted(list(glob.glob(f"{args.world_directory}/*label*.usd")))
    worlds = list(zip(rgb_usds, label_usds))

    init_world()
    terrain_prim = load_mesh( terrain_prim_path, worlds[world_index][0] )
    label_prim   = load_mesh( label_prim_path,   worlds[world_index][1] )

    stage = omni.usd.get_context().get_stage()
    label_material_prim = stage.GetPrimAtPath( label_material_prim_path )
    label_material_prim.CreateAttribute("inputs:specular_level", Sdf.ValueTypeNames.Float)
    label_material_prim.GetAttribute("inputs:specular_level").Set(0.0)
        
    show_rgb(True)
    for ii in range(100):
        kit.update()

def show_rgb(rgb=True):

    global terrain_prim_path, label_prim_path

    stage = omni.usd.get_context().get_stage()
    terrain_prim = stage.GetPrimAtPath( terrain_prim_path )
    label_prim = stage.GetPrimAtPath( label_prim_path )

    if( rgb ):
        terrain_prim.GetAttribute("visibility").Set("inherited")
        label_prim.GetAttribute("visibility").Set("invisible")
    else:
        terrain_prim.GetAttribute("visibility").Set("invisible")
        label_prim.GetAttribute("visibility").Set("inherited")

def randomly_position_camera(bounding_box_mins=(-125, -125, 0), bounding_box_maxes=(125, 125, 150)):
    
    global terrain_prim_path, label_prim_path

    stage = omni.usd.get_context().get_stage()
    bb_cache = bounds_utils.create_bbox_cache()
    label_prim = stage.GetPrimAtPath(label_prim_path)
    terrain_bounding_box = bb_cache.ComputeLocalBound(label_prim).GetRange()

    position_scalar = 1.50
    target_scalar   = 0.75

    minimum = terrain_bounding_box.min
    maximum = terrain_bounding_box.max
    
    #target_position_min = target_scalar * numpy.array([minimum[0], 0, minimum[2]])
    #target_position_max = target_scalar * numpy.array([maximum[0], 0, maximum[2]])
    target_position_min = target_scalar * numpy.array([minimum[0], minimum[2], 0])
    target_position_max = target_scalar * numpy.array([maximum[0], maximum[2], 0])
    target_position = numpy.random.uniform(target_position_min, target_position_max)
    
    minimum_height = maximum[1] * 2 
    maximum_height = maximum[1] * 3
    
    height = numpy.random.uniform(minimum_height, maximum_height)
    radius = numpy.random.uniform(0, height)
    angle  = numpy.random.uniform(0, 2*numpy.pi)
    camera_offset = numpy.array([radius * numpy.cos(angle), radius * numpy.sin(angle), height])
    camera_position = target_position + camera_offset
    
    #camera_position_min = position_scalar * numpy.array([minimum[0], minimum_height, minimum[2]])
    #camera_position_max = position_scalar * numpy.array([maximum[0], maximum_height, maximum[2]])


    #random_position = numpy.random.uniform(bounding_box_mins, bounding_box_maxes)
    #target_position = numpy.random.uniform([0, 0, 0], [0, 0, 0])

    #random_position = numpy.random.uniform(camera_position_min, camera_position_max)
    set_camera_view(eye=camera_position, target=target_position)

# create a custom writer
class MyCustomWriter(Writer):
    def __init__(
        self,
        output_dir,
        rgb = True,
        normals = False,
        starting_frame_id = 0
    ):
        
        print("initialized data writer!")

        self.version = "0.0.1"
        self.backend = BackendDispatch({"paths": {"out_dir": output_dir}})
        if rgb:
            self.annotators.append(AnnotatorRegistry.get_annotator("rgb"))
        if normals:
            self.annotators.append(AnnotatorRegistry.get_annotator("normals"))
        self._frame_id = starting_frame_id + 1 # this is a bit annoying tbh but idk

        self.first_iteration = True
        
        viewport_render_product = rep.create.render_product("/OmniverseKit_Persp", (512, 512))

        self.attach(viewport_render_product)

    def write(self, data: dict):

        print(f"{self._frame_id=}")
        image_filename = f"{self.backend.output_dir}/image_{self._frame_id:05}.png"
        label_filename = f"{self.backend.output_dir}/label_{self._frame_id:05}.png"

        if( self.first_iteration ):
            
            print(f"[{self._frame_id}] Writing {image_filename} ...")
            self.backend.write_image(f"{image_filename}", data["rgb"])

            show_rgb(False)

        else:
            
            print(f"[{self._frame_id}] Writing {label_filename} ...")

            label_data = data["rgb"]
            label_data = label_data[label_data > 0.5]
            label_data = Image.fromarray(data["rgb"])
            label_data = label_data.convert("L", dither=Image.NONE)
            threshold = 128
            label_data = label_data.point(lambda x: 255 if x > threshold else 0, mode='1')

            self.backend.write_image(f"{label_filename}", label_data)
            
            show_rgb(True)

        if( not self.first_iteration ):

            randomly_position_camera()

            self._frame_id += 1

        self.first_iteration = not self.first_iteration

    def on_final_frame(self):
        self._frame_id = 0

terrain_prim_path = "/World/terrain"
label_prim_path = "/World/label"
label_material_prim_path = "/World/label/Looks/material_0/material_0"

print(f"parsing {args.world_directory=}")
rgb_usds = sorted(list(glob.glob(f"{args.world_directory}/*rgb*.usd")))
label_usds = sorted(list(glob.glob(f"{args.world_directory}/*label*.usd")))
worlds = list(zip(rgb_usds, label_usds))
print(f"{worlds=}")
output_directory = f"{args.world_directory}/dataset"

WriterRegistry.register(MyCustomWriter)

time_deltas = []

data_points_per_world = 20
for world_index in range(len(worlds)):
#for world_index in range(0):

    start_time = datetime.now()

    load_world(world_index)
    #writer = MyCustomWriter("/home/joshua/datasets/test1", starting_frame_id=data_points_per_world * world_index)
    writer = MyCustomWriter(output_directory, starting_frame_id=data_points_per_world * world_index)

    print(f"{world_index=}")
    rep.settings.carb_settings("/omni/replicator/RTSubframes", 4)
    rep.orchestrator.run_until_complete(num_frames=2*data_points_per_world)

    clear_world();

    end_time = datetime.now()
    time_delta = (end_time - start_time).total_seconds()

    print(f"\n\n\n\n\ntime to load world, generate {data_points_per_world} data points, and reset: {time_delta}\n\n\n")
    time_deltas.append(time_delta)

average_time_delta = sum(time_deltas) / data_points_per_world
average_time = average_time_delta / len(worlds)

print("\n\n\n\n\n")
print(f"average time: {average_time} s per data point")
print(f"{1/average_time} data points per second")
print("\n\n\n\n\n")

kit.close()
