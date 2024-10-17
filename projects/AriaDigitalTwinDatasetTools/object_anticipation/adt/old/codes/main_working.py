# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from math import tan
from typing import Dict, Set

import numpy as np
import rerun as rr
import os                                   # added by Petros
from collections import deque, defaultdict  # added by Petros
from typing import Dict, List, Tuple, Deque # added by Petros

from itertools import product               # added by Petros ()

import logging
import os
import csv
import json

from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection 

from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.calibration import CameraCalibration, DeviceCalibration
from projectaria_tools.core.sophus import SE3
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.projects.adt import (
    AriaDigitalTwinDataPathsProvider,
    AriaDigitalTwinDataProvider,
    AriaDigitalTwinSkeletonProvider,
    bbox3d_to_line_coordinates,
    DYNAMIC,
    STATIC,
)

from tqdm import tqdm    

from helpers import write_to_excel # Me: added by Petros and is only for debugging

from projectaria_tools.utils.rerun_helpers import (
    AriaGlassesOutline,                                      # Me: Return a list of points to be used to draw the outline of the glasses (line strip).
    ToTransform3D                                            # Me: Helper function to convert Sophus SE3D pose to a Rerun Transform3D
)                                                               

from visualization.rr import (                               # Me: added by Petros
    initialize_rerun_viewer,                                 # Me: Initialize the rerun software    
    log_camera_calibration,                                  # Me: Log the camera features
    log_aria_glasses,                                        # Me: Log the aria glasses
    set_rerun_time,                                           
    process_and_log_image,
    log_device_transformations,                             
    log_dynamic_object,
    log_object,                                              # Me: Log an object 
    log_object_line,                                         # Me: Log an object Line 
    log_vector,                                              # Me: Log the velocity line
    clear_logs_names,                                        # Me: At each timestep clear the objects from the visualization tool 
    clear_logs_ids,
)

from utils.tools import (
    transform_point,                                          # Me: Transformation point from scene to camera frame
    visibility_mask,                                          # Me: Check which points are visible and which are not visible
    time_to_interaction,                                      # Me: Time to interact with each object
    exponential_filter,                                       # Me: Filter the velocity with exponential mean average 
    object_within_radius,                                     # Me: Check the objects that are close to a user
    user_movement_calculation,                                # Me: User's movement calculation
    calculate_relative_pose_difference,                        
    check_point_in_3D_bbox_of_object_position                 # Me: Check where the centroid is located            
)

from utils.openai_models import ( 
    activate_llm,                                             # Me: Query the LLM
    setup_logger,                                             # Me: Setup the logger
    append_to_history_string,                                 # Me: Write the history in a string
    process_llm_response,                                     # Me: Post processing of LLM output 
                                               
)

from utils.objectsGroup_user import (
    ObjectGroupAnalyzer                                       # Me: Class to analyse the objects around the user and specify if the user is changing areas
)

from utils.stats import (
    Statistics                                                # Me: Keep statistics for high dot value and low distance
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_path", type=str, required=True, help="path to the ADT sequence")
    parser.add_argument("--device_number", type=int, default=0, help="Device_number you want to visualize, default is 0")
    parser.add_argument("--down_sampling_factor", type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument("--jpeg_quality", type=int, default=75, help=argparse.SUPPRESS)
    parser.add_argument("--rrd_output_path", type=str, default="", help=argparse.SUPPRESS  )                                  # Me: If this path is set, we will save the rerun (.rrd) file to the given path
    parser.add_argument("--use_llm", action='store_true',help="If you include it in arguments becomes True")                              # Me: added by Petros, if there is a value that 
    parser.add_argument("--runrr", action='store_true',help="Run the the visualization part..same as above")   
    parser.add_argument("--visualize_objects", action='store_true',help="Visualize the objects in the rerun.io")   
    return parser.parse_args()

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
main_logger = logging.getLogger(__name__)

# ==============================================
# Parameters Settting
# ==============================================

time_thresholds = [2] # [1, 2, 3, 4, 5]          # Me: how many seconds before interaction activate LLM
avg_dot_threshold_highs = [0.7]                  # Me: To filter out visible objects. Keep only those that have average dot value higher than this
avg_dot_threshold_lows = [0.2]                      
avg_distance_threshold_highs = [3]               # Me: To filter out visible objects. Keep only those that have average distance value higher than this 
avg_distance_threshold_lows = [1]                   
high_dot_thresholds = [0.9]                      # Me: Count only for these values
distance_thresholds = [2]                        
low_distance_thresholds = [1]
high_dot_counters_threshold = [45]               # Me: Keep only objects with values with counters more than these values
low_distance_counters_threshold = [30]           
time_counters_threshold = [60]                   # Me: 
variables_window_times = [3.0]

# Generate all combinations of the parameters
param_combinations = [
    {
        "time_threshold": t,
        "avg_dot_high": adh,
        "avg_dot_low": adl,
        "avg_distance_high": adhg,
        "avg_distance_low": adlg,
        "high_dot_threshold": hdt,
        "distance_threshold": dt,
        "low_distance_threshold": ldt,
        "high_dot_counters_threshold": hdct,
        "low_distance_counters_threshold": ldct,
        "time_counters_threshold": tc,
        "window_time": w
    }
    for t, adh, adl, adhg, adlg, hdt, dt, ldt, hdct, ldct, tc, w in product(
        time_thresholds, avg_dot_threshold_highs, avg_dot_threshold_lows, 
        avg_distance_threshold_highs, avg_distance_threshold_lows,
        high_dot_thresholds, distance_thresholds, low_distance_thresholds, 
        high_dot_counters_threshold, low_distance_counters_threshold, time_counters_threshold, 
        variables_window_times
    )
]

# ==============================================
# Run for different parameter combinatons
# ==============================================

for parameters in param_combinations: # TODO parallel 4 loop
        
    def main():
        
        work_in_xz_plane = True
        
        # ==============================================
        # Filenames / Paths  & Load of the Data
        # ==============================================
        args = parse_args()
        
        # Base folder path for saving predictions
        project_path = "Documents/projectaria_sandbox/projectaria_tools/projects/AriaDigitalTwinDatasetTools/object_anticipation/adt/"
        sequence_path = args.sequence_path

        # Datasets path
        datasets_path = 'Documents/projectaria_tools_adt_data/'
        dataset_folder = os.path.join(datasets_path, sequence_path)
        os.makedirs(dataset_folder, exist_ok=True)                                          # Me: Ensure the entire directory 
        
        # VRS file and Ground 
        vrsfile = os.path.join(dataset_folder, "video.vrs")
        ADT_trajectory_file = os.path.join(dataset_folder, "aria_trajectory.csv")
        
        # Path to log items for the LLM - Define the CSV file to log the items and check if it exists to write the header
        csv_file = os.path.join(project_path,'utils','txt_files','interaction_log.csv')

        # Save the list to a file
        json_folder = os.path.join(project_path,'utils','json')
        os.makedirs(json_folder, exist_ok=True)                        
        json_file = os.path.join(json_folder,'param_combinations.json')
        
        with open(json_file, 'w') as file:
            json.dump(param_combinations, file)
            
        # Parameters folder name
        parameter_folder_name = (
                f"time_{parameters['time_threshold']}_"
                f"highdot_{parameters['high_dot_threshold']}_"
                f"highdotcount_{parameters['high_dot_counters_threshold']}_"
                f"dist_{parameters['distance_threshold']}_"
                f"distcount_{parameters['low_distance_counters_threshold']}"
            )
    
        # Print the paths
        print("Sequence_path: ", dataset_folder)
        print("Project_path", project_path)
        print("VRS File Path: ", vrsfile)  
        print("GT trajectory path: ", ADT_trajectory_file)
        
        try:
            paths_provider = AriaDigitalTwinDataPathsProvider(dataset_folder) 
            data_paths = paths_provider.get_datapaths_by_device_num(args.device_number)
            gt_provider = AriaDigitalTwinDataProvider(data_paths)
        except Exception as e:
            print("Error: ", str(e))
            exit(-1)

        # True to run the rerun.io 
        args.runrr and initialize_rerun_viewer(rr, args)

        # Load the device trajectory timestamps
        aria_pose_start_timestamp = gt_provider.get_start_time_ns()                     # Me: Get the start time of the Aria poses in nanoseconds
        aria_pose_end_timestamp = gt_provider.get_end_time_ns()                         # Me: Get the end time
        rgb_stream_id = StreamId("214-1")
        
        # Load the camera calibration
        rgb_camera_calibration = gt_provider.get_aria_camera_calibration(rgb_stream_id) # Me: Get the camera calibration of an Aria camera, including intrinsics, distortion params,and projection functions.
        T_Device_Cam = rgb_camera_calibration.get_transform_device_camera()             # Me: Τhis does not change based on time
        args.runrr and log_camera_calibration(rr, rgb_camera_calibration, args)
        
        # Get all timestamps (in ns) of all observations of an Aria sensor
        img_timestamps_ns = gt_provider.get_aria_device_capture_timestamps_ns(rgb_stream_id)    
        img_timestamps_ns = [
            img_timestamp_ns 
            for i, img_timestamp_ns in enumerate(img_timestamps_ns)
            if (
                img_timestamp_ns >= aria_pose_start_timestamp
                and img_timestamp_ns <= aria_pose_end_timestamp
            )
        ]
        start_time = img_timestamps_ns[0] / 1e9
        
        # Log Aria Glasses outline
        raw_data_provider_ptr = gt_provider.raw_data_provider_ptr()
        device_calibration = raw_data_provider_ptr.get_device_calibration()
        aria_glasses_point_outline = AriaGlassesOutline(device_calibration)
        args.runrr and log_aria_glasses(rr, aria_glasses_point_outline)
        
        # ==============================================
        # Initialization 
        # ==============================================
        
        dynamic_obj_pose_cache: Dict[str, SE3] = {}                                     # Me: initializes a dictionary maps string keys (likely object identifiers) to SE3 objects
        static_obj_ids: Set[int] = set()                                                # Me: initializes a set intended to store the IDs of static objects. 
        dynamic_obj_moved: Set[str] = set()                                             # Me: initializes a set intended to store the IDS of dynamic objects.

        # Intialize variable for visualizations part 
        previous_obj_ids = set()                                                        # Me: keep track of previously logged objects ids 
        previous_obj_names = set()                                                      # Me: keep track of previously logged objects names
        
        # Collection for the time window variables
        average = True                                                                  # Me: use the accumulated average dot value and distance to filter the objects
        previous_time_ns = aria_pose_start_timestamp                                    # Me: Initialize time to store duration of each object
        past_dots: Dict[int, List[Tuple[int, float]]] = defaultdict(list)               # Me: Past dot products over a period of seconds
        past_distances: Dict[int, List[Tuple[int, float]]] = defaultdict(list)          # Me: Past distances over a period of seconds
        visibility_counter: Dict[int, List[Tuple[int, int]]] = defaultdict(list)        # Me: List the visibility counters over a period of seconds
        visibility_duration: Dict[int, List[Tuple[int, float]]] = defaultdict(list)     # Me: List the visibility duration over a period of seconds
        avg_dots: Dict[int, float] = defaultdict(float)                                 # Me: Accumulated / Average dot over a period seconds
        avg_distances: Dict[int, float] = defaultdict(float)                            # Me: Averaged Distances over a period of seconds
        high_dot_counts: Dict[int, List[Tuple[int, float]]] = defaultdict(list)         # Me: High dot streaks over a period of seconds
        close_distance_counts: Dict[int, List[Tuple[int, float]]] = defaultdict(list)   # Me: Close distance over a period of seconds
        
        # Activation of LLM
        llm_activated = False                                                           # Me: LLM activated or not
        # history_log: Deque[Dict] = deque()                                            # Me: Initialize the history log as a deque. This was needed when we put the old values 
        history_log: Dict                                                               # Me: This is for now 
        predictions = []                                                                # Me: Predictions of the LLM      
        predictions_dict = {}                                                           # Μe: Predictions of the LLM as a dictionary with the timestamps
        goals_dict = {}
        llm_times = []                                                                  # Me: Timestamps that LLM predicted each output
        user_velocity_before = None                                                     # Me: Initialize previous velocity
        user_position_before = None                                                     # Me: Calculate the user's position
        objects_within_radius = []                                                      # Me: Objects in the vicinity of the user
        previous_objects_within_radius = []                                             # Me: Is used to check if the user is stll in the same area that LLM has been activated in order to avoid reactivatiomn of the LLM 
        last_activation_time = 0                                                        # Me: the time activatiom of an LLM
        all_unique_object_names_with_high_dot = set()                                               
    
        # Initialize classes
        group_analyzer = ObjectGroupAnalyzer(history_size=15, future_size=5, change_threshold=0.7)                                         
        statistics = Statistics(
                                parameters['window_time'], 
                                parameters["high_dot_threshold"], 
                                parameters["distance_threshold"], 
                                parameters["low_distance_threshold"], 
                                parameters["time_threshold"]
                                )      # Me: Initialize the ObjectStatistics instance
        
        # ==============================================
        # Load the Ground truth data 
        # ==============================================
        
        with open(os.path.join(project_path,'data','gt',args.sequence_path,'movement_time_dict.json'), 'r') as json_file:
            movement_time_dict = json.load(json_file)
        
        gt_object_names = np.array(list(movement_time_dict.keys()))
        gt_start_times = np.array([movement_time_dict[obj]['start_time'] for obj in gt_object_names])
        gt_end_times = np.array([movement_time_dict[obj]['end_time'] for obj in gt_object_names])
        
        # ==============================================
        # Loop over all timestamps in the sequence
        # ==============================================

        for timestamp_ns in tqdm(img_timestamps_ns):
            args.runrr  and set_rerun_time(rr, timestamp_ns)
        
            ## Current time in seconds
            current_time_ns = timestamp_ns
            current_time_s = round((current_time_ns / 1e9 - start_time), 3)
            
            if current_time_s > 70 and current_time_s < 71 :
                print('stop')      
                  
            ## Time Difference
            time_difference_ns = (current_time_ns - previous_time_ns) / 1e9                                                  # Me: Calculate the time difference in seconds
            previous_time_ns = current_time_ns

            ## Clear previously logged objects and lines
            if args.runrr:
                clear_logs_ids(rr, previous_obj_ids)
                clear_logs_names(rr, previous_obj_names)
            
            previous_obj_ids.clear()
            previous_obj_names.clear()
            
            ## Log RGB image
            image_with_dt = gt_provider.get_aria_image_by_timestamp_ns(timestamp_ns, rgb_stream_id)
            args.runrr and process_and_log_image(rr, args, image_with_dt)
            
            # ==============================================
            # Users poses - position / velocity / movement (scene)
            # ==============================================                                                          
                                                                                    
            aria_3d_pose_with_dt = gt_provider.get_aria_3d_pose_by_timestamp_ns(timestamp_ns)                                 # Me: Pose of the device

            if aria_3d_pose_with_dt.is_valid():
                p_dev_scene = aria_3d_pose_with_dt.data()                                                                     # Me: Pose of the user's device on Scene frame
                T_Scene_Device = p_dev_scene.transform_scene_device                                                           # Me: SE3 of the device 
                T_Scene_Cam = T_Scene_Device @ T_Device_Cam                                                                   # Me: SE3 from Camera to Scene
                   
                ## User's position and velocity in the scene frame 
                user_position_scene = aria_3d_pose_with_dt.data().transform_scene_device.translation()[0]      
                user_velocity_device = aria_3d_pose_with_dt.data().device_linear_velocity # given in the device frame as per https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/slam/mps_trajectory
                
                # VELOCITY - From Device to Scene (only rotation is necessary) 
                """
                    Take only the ROTATION / Velocity vectors represent rates of change of position, so they are not anchored to a specific position in space
                    
                    3 ways to do it
                    
                    1. user_velocity_scene    = T_Scene_Device.rotation().to_matrix() @ user_velocity_device
                    2. user_velocity_scene_v2 = (T_Scene_Device @ user_velocity_device).reshape(1,3)[0] - user_position_scene
                    3. user_velocity_scene_v3 = (T_Scene_Device.to_matrix() @ np.append(user_velocity_device, [1]))[0:3] - user_position_scene
                """ 
                user_velocity_scene = T_Scene_Device.rotation().to_matrix() @ user_velocity_device
                
                #  EXPONENTIAL MOVEMENT AVERAGE (EMA) & TOTAL MOVEMENT
                if user_position_before is None and user_velocity_before is None:
                    
                    # USER/DEVICE POSITION 
                    user_ema_position = user_position_before = user_position_scene  
                    user_ema_velocity = user_velocity_before = user_velocity_scene   
                    
                    # Initialise user's movement 
                    user_total_movement = 0                                                                                  # Me: Total movement along the sequene (lenght of trajectory)
                    user_relative_total_movement = 0                                                                         # Me: Is used to activate LLM again

                    # USER/DEVICE POSITION at each timestep
                    users_position = deque([(current_time_s, 0)])           
                    users_velocity = deque([(current_time_s, 0)])     
                    
                else:
                    # EXPONENTIAL MOVEMENT AVERAGE POSITION AND AVERAGE 
                    user_ema_position = exponential_filter(user_position_scene, user_position_before, alpha = 0.9)            # Me: Exponential filter for the position to reduce the noise 
                    user_ema_velocity = exponential_filter(user_velocity_scene, user_velocity_before, alpha = 0.9)            # Me: Apply exponential filter 
                    
                    # MOVEMENT 
                    user_movement_timestep = user_movement_calculation(user_position_before, user_ema_position) 
                    user_total_movement += user_movement_timestep
                    user_relative_total_movement += user_movement_timestep
                    
                    # DEQUE POSITION & DEQUE VELOCITY
                    users_position.append((current_time_s, user_ema_position))                                                 
                    users_velocity.append((current_time_s, user_ema_velocity))       
                    
                    # recalculate the position & velocity before
                    user_position_before = user_ema_position
                    user_velocity_before = user_ema_velocity
                    
                ## Insert this info in the runio.rr software
                args.runrr and log_device_transformations(rr, p_dev_scene, T_Device_Cam, ToTransform3D) 
            
            # ==============================================
            # Objects Poses
            # ==============================================      
            
            bbox3d_with_dt = gt_provider.get_object_3d_boundingboxes_by_timestamp_ns(timestamp_ns)
            assert bbox3d_with_dt.is_valid(), "3D bounding box is not available"
            bboxes3d = bbox3d_with_dt.data()                                                                                        # Me: Objects data

            # TODO: check where the centroid is located
            
            ## Extract object IDs and their positions
            obj_ids = np.array(list(bboxes3d.keys()))                                                                                # Me: Ids of the objects
            obj_names = np.array([gt_provider.get_instance_info_by_id(obj_id).name for obj_id in obj_ids])                           # Me: Names of the objects
            obj_positions_scene = np.array([bbox_3d.transform_scene_object.translation() for bbox_3d in bboxes3d.values()])          # Me: Positions on Scene frame #TODO: maybe add the [0] to take the first elemeent   
            
            # ==============================================
            # Visual Objects in Camera Frame
            # ==============================================  
            
            T_Cam_Scene = T_Scene_Cam.inverse()                                                                               # Me: Transform positions to the camera frame
            
            ## Objects in Camera Frame
            obj_positions_cam = np.array([transform_point(T_Cam_Scene, pos.reshape(1, 3)) for pos in obj_positions_scene])    # Me: Positions on Camera frame
            obj_positions_cam_reshaped = obj_positions_cam.reshape(-1, 3, 1)                                                  # Me: Prepare positions for projection
            
            ## Produce the visible mask to work only with objects that are in the camera frame
            valid_mask = visibility_mask(obj_positions_cam_reshaped, rgb_camera_calibration)                                  # Me: Produce the Mask
            
            ## Take the only poses of the visible objects   
            T_scene_object = {}
            for key, (include, value) in zip(bboxes3d.keys(), zip(valid_mask, bboxes3d.values())):                            # Me: Filter the dictionary based on the boolean array
                if True: # if include: to take only the visible objects
                    T_scene_object[key] = value.transform_scene_object
            
            ## Take ids, names, position and distances for the visible objects 
            object_ids = visible_obj_ids = obj_ids[valid_mask]                                                                # Me: Filter the ids of visible objects
            visible_obj_names = obj_names[valid_mask]                                                                         # Me: Filter the names of visible objects
            object_positions = visible_obj_positions_scene = obj_positions_scene[valid_mask]                                  # Me; Filter the positions visible objects in scene 
            visible_obj_positions_cam = obj_positions_cam[valid_mask]                                                         # Me: Filter the positions visible objects in camera
            
            # ==============================================
            # Camera and Device Axes Transformation in Scene and World Frames
            # ==============================================  
            
            # Device and Camera
            device_x_axis = cam_x_axis = np.array([1, 0, 0]).reshape(3, 1)
            device_y_axis = cam_y_axis = np.array([0, 1, 0]).reshape(3, 1)
            device_z_axis = cam_z_axis = np.array([0, 0, 1]).reshape(3, 1)
            
            # CAMERA coordinates in the scene frame (having the traslation into account)
            cam_x_axis_scene = (T_Scene_Cam @ cam_x_axis).reshape(1,3)[0]  
            cam_y_axis_scene = (T_Scene_Cam @ cam_y_axis).reshape(1,3)[0]
            cam_z_axis_scene = (T_Scene_Cam @ cam_z_axis).reshape(1,3)[0] # these value points the end of z axis. so from the origin has an opposite direction to where the user is looking and that's why the dot value is negative
            
            # CAMERA coordinates in the scene frame (having the traslation into account)
            cam_x_axis_rotation = (T_Scene_Cam.rotation().to_matrix() @ cam_x_axis)[:,0]
            cam_y_axis_rotation = (T_Scene_Cam.rotation().to_matrix() @ cam_y_axis)[:,0]
            cam_z_axis_rotation = (T_Scene_Cam.rotation().to_matrix() @ cam_z_axis)[:,0]
            
            # DEVICE coordinates in the scene frame
            device_x_axis_scene = (T_Scene_Device @ device_x_axis).reshape(1,3)[0]
            device_y_axis_scene = (T_Scene_Device @ device_y_axis).reshape(1,3)[0]
            device_z_axis_scene = (T_Scene_Device @ device_z_axis).reshape(1,3)[0]
            
            # WORLD coordinates 
            world_x_axis = np.array([1,0,0])
            world_y_axis = np.array([0,1,0])
            world_z_axis = np.array([0,0,1])
            
            # ==============================================
            # Dot Products - Distances
            # ==============================================  
            
            # VECTORS FROM CAMERA AND DEVICE TO OBJECTS
            camera_position_scene = T_Scene_Cam.translation()                                                                                    # Me: Camera position in scene
            vector_camera_objects_scene = obj_positions_scene[:, 0] - camera_position_scene                                     
            vector_devive_objects_scene = obj_positions_scene[:, 0] - user_position_scene
            
            # CALCULATE DOT PRODUCT IN 2D
            if work_in_xz_plane:
                
                # PROJECT VECTORS ONTO XZ PLANE (ignore Y component)
                vector_camera_objects_scene_xz = np.copy(vector_camera_objects_scene)
                vector_camera_objects_scene_xz[:, 1] = 0  # Set Y component to 0
                unit_vector_camera_objects_scene_xz = vector_camera_objects_scene_xz / np.linalg.norm(vector_camera_objects_scene_xz, axis=1, keepdims=True)

                # PROJECT CAMERA Z AXIS ONTO XZ PLANE (ignore Y component)
                cam_z_axis_rotation_xz = np.copy(cam_z_axis_rotation)
                cam_z_axis_rotation_xz[1] = 0  # Set Y component to 0

                # Normalize the camera Z axis vector on the XZ plane
                cam_z_axis_rotation_xz /= np.linalg.norm(cam_z_axis_rotation_xz)

                # DOT PRODUCT IN XZ PLANE
                dot_products_array = np.dot(unit_vector_camera_objects_scene_xz, cam_z_axis_rotation_xz)

            else:
                unit_vector_camera_objects_scene = vector_camera_objects_scene / np.linalg.norm(vector_camera_objects_scene, axis=1, keepdims=True)  # Me: Normalize the vectors
            
                # DOT PRODUCT
                dot_products_array = np.dot(unit_vector_camera_objects_scene, cam_z_axis_rotation)                                              
            
            all_dot_products = dot_products_array.tolist()                                                                                                 # Me: Dot Product (camera z axis / camera to object vector
                
            # DISTANCES 
            distance_camera_objects_scene = np.linalg.norm(vector_camera_objects_scene, axis=1)                      
            distance_device_objects_scene = np.linalg.norm(vector_devive_objects_scene, axis=1)
            all_distances = distance_device_objects_scene.tolist()  
            
            # VISIBLE VECTORS FROM CAMERA AND DEVICE TO OBJECTS
            visible_vector_camera_objects_scene = visible_obj_positions_scene[:, 0] - camera_position_scene                                     
            visible_vector_devive_objects_scene = visible_obj_positions_scene[:, 0] - user_position_scene
            visible_unit_vector_camera_objects_scene = visible_vector_camera_objects_scene / np.linalg.norm(visible_vector_camera_objects_scene, axis=1, keepdims=True)  # Me: Normalize the vectors

            # VISIBLE DOT PRODUCT
            visible_dot_products = dot_products_array[valid_mask]                                        
            dot_products = visible_dot_products.tolist()   
            
            # VISIBLE DISTANCES
            visible_distance_camera_objects_scene = np.linalg.norm(visible_vector_camera_objects_scene, axis=1)                      
            visible_distance_device_objects_scene = np.linalg.norm(visible_vector_devive_objects_scene, axis=1)                     
            distances =  visible_distance_device_objects_scene.tolist()                                                                            # Me: Filter the visible objects in scene

            # ==============================================
            # Time Window - Accumulated / Average Values & Counts
            # ==============================================  
            
            (avg_dots, 
            avg_distances, 
            past_dots, 
            past_distances, 
            visibility_counter, 
            visibility_duration, 
            high_dot_counts, 
            close_distance_counts,
            very_close_distance_counts,
            time_to_approach_counts, 
            object_time_interaction, 
            avg_dots_list, 
            avg_distances_list) = statistics.time_window(
                                                        current_time_s, 
                                                        time_difference_ns, 
                                                        user_ema_position, 
                                                        user_velocity_device,
                                                        object_ids, 
                                                        object_positions,
                                                        dot_products, 
                                                        distances,
                                                        T_Scene_Device
                                                    )

            # ==============================================
            # Filter Visible Objects with Average Values
            # ==============================================  
            
            # MASK
            if average == True:
                
                # high dot mask and relaxed distance
                high_dot_mask = np.array([avg_dots[obj_id] > parameters["avg_dot_high"] for obj_id in visible_obj_ids])                      # Me: high dot mask shape for those objects that have high accummulated dot
                high_distance_mask = np.array([avg_distances[obj_id] < parameters["avg_distance_high"] for obj_id in visible_obj_ids])       # Me: high distance mask
                
                # low average dot mask but also low distance
                low_dot_mask = np.array([avg_distances[obj_id] > parameters["avg_dot_low"] for obj_id in visible_obj_ids])                   # Me: low dot mask
                low_distance_mask = np.array([avg_distances[obj_id] < parameters["avg_distance_low"] for obj_id in visible_obj_ids])         # Me: low distance mask (because the minimum distace is 0.56 from the object)
                
                # combined masks
                combined_high_high_mask = high_dot_mask & high_distance_mask                                       
                combined_low_low_mask = low_dot_mask & low_distance_mask                                        
                combined_mask = combined_high_high_mask | combined_low_low_mask                           
                
            else:                                                                                              
                
                # high dot mask and relaxed distance
                high_dot_mask = dot_products > 0.8                                                                      # Me: high dot mask shape is shape (189,) 1D array
                high_distance_mask = distances < 3                                                                      # Me: high distance mask
                
                # low average dot mask but also low distance
                low_dot_mask = dot_products > 0.2                                                                       # Me: low dot mask 
                low_distance_mask = distances < 0.9                                                                     # Me: low distance mask (because the minimum distace is 0.56 from the object)
                
                # combined masks
                combined_high_high_mask = high_dot_mask & high_distance_mask                                            # Me: if the distance is high the dot should be high to accept 
                combined_low_low_mask = low_dot_mask & low_distance_mask                                                # Me: if the dot is low the distance should be low to accept 
                combined_mask = combined_high_high_mask | combined_low_low_mask

            # IDs, NAMES
            filtered_obj_ids = visible_obj_ids[combined_mask]
            filtered_obj_names = visible_obj_names[combined_mask]
            filtered_vector_camera_objects_scene = visible_vector_camera_objects_scene[combined_mask]
            
            # POSITIONS
            filtered_obj_positions_scene = visible_obj_positions_scene[combined_mask]
            filtered_obj_positions_cam = visible_obj_positions_cam[combined_mask]
            
            # DOT PRODUCTS
            filtered_dot_products = visible_dot_products[combined_mask]
            filtered_names_dot = {gt_provider.get_instance_info_by_id(obj_id).name: filtered_dot_products[i] for i, obj_id in enumerate(filtered_obj_ids)}
            
            # DISTANCES 
            filtered_distances_cam = visible_distance_camera_objects_scene[combined_mask]
            filtered_distances = visible_distance_device_objects_scene[combined_mask]
            filtered_names_distances = {gt_provider.get_instance_info_by_id(obj_id).name: filtered_distances[i] for i, obj_id in enumerate(filtered_obj_ids)}   
            
            # VISIBILITY COUNTER & DURATION                   
            filtered_counter = {obj_id: visibility_counter[obj_id] for obj_id in filtered_obj_ids if obj_id in visibility_counter}     
            filtered_duration = {obj_id: visibility_duration[obj_id] for obj_id in filtered_obj_ids if obj_id in visibility_duration}
            
            # HIGH DOT / LOW DISTANCE / TIME COUNTERS 
            filtered_high_dot_counts = {obj_id: high_dot_counts[obj_id] for obj_id in filtered_obj_ids if obj_id in high_dot_counts}
            filtered_low_distance_counts = {obj_id: close_distance_counts[obj_id] for obj_id in filtered_obj_ids if obj_id in close_distance_counts}
            filtered_time_to_approach_counts = {obj_id: time_to_approach_counts[obj_id] for obj_id in filtered_obj_ids if obj_id in time_to_approach_counts}
                   
            # ==============================================
            # Keep the Important Context Information for the feasible objects
            # ==============================================  
            
            # HIGH DOT COUNTS / LOW DISTANCE COUNTS / TIME COUNTS - DICTIONARIES {NAME: COUNT}
            filtered_names_len_high_dot_counts = {gt_provider.get_instance_info_by_id(obj_id).name: len(high_dot_counts[obj_id]) for obj_id in filtered_obj_ids if obj_id in high_dot_counts}
            filtered_names_len_low_distance_counts = {gt_provider.get_instance_info_by_id(obj_id).name: len(close_distance_counts[obj_id]) for obj_id in filtered_obj_ids if obj_id in close_distance_counts}
            filtered_names_len_time_counts = {gt_provider.get_instance_info_by_id(obj_id).name: len(time_to_approach_counts[obj_id]) for obj_id in filtered_obj_ids if obj_id in time_to_approach_counts}

            # TIME TO INTERACTION 
            filtered_object_time_interaction = {gt_provider.get_instance_info_by_id(obj_id).name: object_time_interaction[obj_id] for obj_id in filtered_obj_ids} 
            
            # ==============================================
            # 4 Criteria to enable LLM (1. High dot products duration 2. Low distance duration 3. Time to contact 4. High visibility duration)
            # ==============================================  
            
            """
            
            1. Initialize dictionaries and lists to be used for the LLM activation 
                - We want to have objects that presents high dot values consistently 
                - we want to have objects that are close to the user consistently
                - We want to have objects that are approachable in less than 2 seconds (this list is a subset of the above list)
                - We want to have objects that are visible to the user's camera for a significant amount of time over 2 seconds 
                - We want objects that have shown high dot history in the past 
            
            2. Assummption: 
                - User will interact with objects that are consistently in the user's focus
                - User will interact with objects that are close to him 
                - User will interact with objects is close to one of these in less a certain amount of time (e.g. 2 seconds)
                - User will interact with objects that have been seen consistently (so for this reason we need to add the history)
                - User will interact with objects that are visible to the user for a certain amount of time 
                
            2. Store the objects 
                - High dot counts > 45 counts or 1.5 seconds than 3
                - Close distance < 30 counts or 1 second from the 3
                - Time to approach
                
            """
            # At each frame needs to updates thes following dictionaries / lists to satisfy our criteria
            
            # high dot values & counts
            objects_names_high_dot_values = {}    
            objects_names_high_dot_counts = {}
            
            # distance values & counts
            objects_names_low_distance_values = {}
            objects_names_low_distance_counts = {}
                
            # time to approach dictionaries 
            objects_time_approach_dict = {}  
            objects_time_approach_list = []      

            # time less than 2 seconds 
            objects_names_less_than_2_seconds_dict = {}
            objects_names_less_than_2_seconds_list = []
            
            # time less than 2 seconds and the counts that is satisfied 
            objects_less_than_2_seconds_with_counts_dict = {}
            objects_less_than_2_seconds_with_counts_list = []
            
            # duration of objects
            objects_high_duration= {}
            
            # Use NumPy to find objects in motion
            objects_in_motion = gt_object_names[(gt_start_times <= current_time_s) & (current_time_s <= gt_end_times)].tolist()
            
            ## Identify objects with HIGH DOT COUNTS    
            """ 
            1. ---> store the high dot counters <----
            """
            for index, object_id in enumerate(filtered_obj_ids):
                if object_id in filtered_high_dot_counts and len(filtered_high_dot_counts[object_id]) >= parameters["high_dot_counters_threshold"]:               
                    object_name = gt_provider.get_instance_info_by_id(object_id).name                                                               
                    if object_name in objects_in_motion:
                        continue
                    
                    # store the object names that satisfy the criteria for high dot
                    objects_names_high_dot_values[object_name] = f"{float(filtered_dot_products[index]):.3f}"                                        
                    objects_names_high_dot_counts[object_name] = filtered_names_len_high_dot_counts[object_name]
                    
            ## Keep the objects that have shown high dot values until the current time (regardless of the number of counts)
            
            """
            1. ---> First part put in the history all the keys that have at least one counter with dot value higher than 0.9 <---
            2. ---> Second part has in the history all the keys that have satisfied the criteria of more than 45 at least once <--- I follow this way
            """
            # all_unique_object_names_with_high_dot |= set(filtered_names_len_high_dot_counts.keys()) 
            all_unique_object_names_with_high_dot |= set(objects_names_high_dot_values.keys()) 
            objects_with_high_dot_history = list(all_unique_object_names_with_high_dot)     
            
            # TODO: time_to_approach={gt_provider.get_instance_info_by_id(object_id).name   }   
                                        
            ## Identify objects with LOW DISTANCE COUNTS
            for index, object_id in enumerate(filtered_obj_ids):
                if object_id in filtered_low_distance_counts and len(filtered_low_distance_counts[object_id]) >= parameters["low_distance_counters_threshold"]:   # Me: the user is around this object for more than 30 frames within 90 frames
                    object_name = gt_provider.get_instance_info_by_id(object_id).name                                                               
                    object_time_xyz, object_time_xz = statistics.interaction_time_user_object(user_velocity_device, user_ema_position, filtered_obj_positions_scene[index][0], T_Scene_Device)
                    
                    if object_time_xz < parameters["time_threshold"]:
                        print(f"\t Time to approach {object_name} is less than 2 seconds and in particular: {object_time_xz}")
                     
                    # DEBUGGING PURPOSES
                    if object_name in ["ChoppingBoard", "KitchenKnife", "WoodenSpoon", "WoodenFork", "Donut_B", "Cereal_Anon", "DinoToy", "WoodenToothbrushHolder", "Cracker_Anon", "BlackCeramicMug"]:
                        
                        # ==============================================
                        # 3D MOTION 
                        # ==============================================
                        
                        # DISTANCE
                      
                        displacement_vector_xyz = (filtered_obj_positions_scene[index][0] - user_ema_position)
                        distance_xyz = np.linalg.norm(displacement_vector_xyz)
                        displacement_unit_vector_xyz = displacement_vector_xyz / np.linalg.norm(displacement_vector_xyz)
                        
                        # VELOCITY 
                        velocity_xyz = T_Scene_Device.rotation().to_matrix() @ user_velocity_device

                        # PROJECTED VELOCITY
                        projected_velocity_xyz = np.dot(velocity_xyz, displacement_unit_vector_xyz) * displacement_unit_vector_xyz 
                        speed_xyz = np.linalg.norm(projected_velocity_xyz)
                        time_xyz = distance_xyz / speed_xyz
                        
                        # ==============================================
                        # PLANAR MOTION ON XZ PLANE
                        # ==============================================

                        # DISTANCE
                        displacement_vector_xz = np.array([displacement_vector_xyz[0], 0, displacement_vector_xyz[2]])
                        displacement_unit_vector_xz = displacement_vector_xz / np.linalg.norm(displacement_vector_xz)
                        distance_xz = np.linalg.norm(displacement_vector_xz)
                        
                        # VELOCITY 
                        velocity_xz = np.array([velocity_xyz[0], 0, velocity_xyz[2]])
                          
                        # ==============================================
                        # 2 WAYS OF CALCULATING THE TIME
                        # ==============================================
                        
                        # 1ST WAY
                        projected_velocity_xz_v1 = np.array([projected_velocity_xyz[0], 0, projected_velocity_xyz[2]])
                        speed_xz_v1 = np.linalg.norm(projected_velocity_xz_v1)
                        time_xz_v1 = distance_xz / speed_xz_v1
                        
                        # 2ND WAY
                        projected_velocity_xz_v2 = np.dot(velocity_xz, displacement_unit_vector_xz) * displacement_unit_vector_xz 
                        speed_xz_v2 = np.linalg.norm(projected_velocity_xz_v2)
                        time_xz_v2 = distance_xz / speed_xz_v2
                        
                        # ==============================================
                        # PRINTS
                        # ==============================================
                        
                        print("-" * 40)  # Print a line of dashes as a separator | # More sophisticated separator)
                        print('')
                        print(f"\t Position of {object_name} in the space is: {filtered_obj_positions_scene[index][0]}")
                        print(f"\t Position of user in the space is: {user_ema_position}")
                        print('')
                        print(f"\t Displacement from user to the {object_name} is : {displacement_vector_xyz}")
                        print(f"\t Distance is {distance_xyz}")
                        print('')
                        print(f"\t Velocity of user in the space is: {velocity_xyz}")
                        print(f"\t Projected Velocity of user in the space is: {projected_velocity_xyz}")
                        print(f"\t Speed is {speed_xyz}")
                        print(f"\t Time is {time_xyz}")
                        print('')
                        print(f"\t Displacement from user to the {object_name} in xz plane is : {displacement_vector_xz}")
                        print(f"\t Distance in 2D is {distance_xz}")
                        print('')
                        print(f"\t Velocity of user in 2D is: {velocity_xz}")  
                        print(f"\t Projected Velocity v1 of user in the space is: {projected_velocity_xz_v1}")
                        print(f"\t Speed in 2D is {speed_xz_v1}")
                        print(f"\t Time is {time_xz_v1}")
                        print('')
                        print(f"\t Projected Velocity v2 of user in the space is: {projected_velocity_xz_v2}")
                        print(f"\t Speed in 2D is {speed_xz_v2}")
                        print(f"\t Time is {time_xz_v2}")
                        print(f"\t Time as calculated using the function of STATS class", {object_time_xz})
                        
                        if args.runrr and args.visualize_objects:
                            
                            # ==============================================
                            # VISUALIZATION
                            # ==============================================
                            
                            # OBJECT LINE
                            log_object_line(rr, gt_provider.get_instance_info_by_id(object_id) , user_ema_position, filtered_obj_positions_scene[index][0])

                            # OBJECT DISTANCE AND DISPLACEMENT
                            log_vector(rr, f"debugging_distance_user_{object_name}", user_ema_position, filtered_obj_positions_scene[index][0])      # line from user to object
                            log_vector(rr, f"debugging_distance_displacement_{object_name}", np.array([0,0,0]), displacement_vector_xyz)             # line from orgin with same direction and magnitude as from user to object
                            log_vector(rr, f"debugging_distance_displacement_{object_name}_xz", np.array([0,0,0]) , displacement_vector_xz)
                            log_vector(rr, f"debugging_distance_unit_displacement_{object_name}", np.array([0,0,0]) , displacement_unit_vector_xyz)
                            log_vector(rr, f"debugging_distance_unit_displacement_{object_name}_xz", np.array([0,0,0]) , displacement_unit_vector_xz)
                            
                            # PROJECTED VELOCITY
                            log_vector(rr, "debugging_projected_velocity", np.array([0,0,0]), projected_velocity_xyz) 
                            log_vector(rr, "debugging_projected_velocity_xz_v1", np.array([0,0,0]), projected_velocity_xz_v1) 
                            log_vector(rr, "debugging_projected_velocity_xz_v2", np.array([0,0,0]), projected_velocity_xz_v2) 
                            
                            # VELOCITY
                            log_vector(rr, "debugging_velocity", np.array([0,0,0]),  (T_Scene_Device.rotation().to_matrix() @ user_velocity_device)) # rotated velocity
                            log_vector(rr, "debugging_velocity_2d", np.array([0,0,0]), np.array([(T_Scene_Device.rotation().to_matrix() @ user_velocity_device)[0], 0, (T_Scene_Device.rotation().to_matrix() @ user_velocity_device)[2]]))
                            log_vector(rr, "debugging_device_velocity", user_ema_position,  (T_Scene_Device @ user_velocity_device)[:,0])
                            
                            # CAMERA Z-AXIS ONLY ROTATION TO SCENE
                            cam_z_axis_rotation = (T_Scene_Cam.rotation().to_matrix() @ cam_z_axis)[:,0]
                            cam_z_axis_rotation_xz =  np.array([cam_z_axis_rotation[0], 0, cam_z_axis_rotation[2]]) 
                            log_vector(rr, "origin_camera_z_axis_rotation_only", np.array([0,0,0]), cam_z_axis_rotation)
                            log_vector(rr, "origin_camera_z_axis_rotation_only_xz", np.array([0,0,0]), cam_z_axis_rotation_xz)
                            log_vector(rr, "origin_camera_z_axis", np.array([0,0,0]), cam_z_axis_scene)
                            
                            # CAMERA
                            log_vector(rr, "camera_x_axis", camera_position_scene[0], cam_x_axis_scene)
                            log_vector(rr, "camera_y_axis", camera_position_scene[0], cam_y_axis_scene)
                            log_vector(rr, "camera_z_axis", camera_position_scene[0], cam_z_axis_scene)
                            
                            # WORLD
                            log_vector(rr, "world_x_axis", np.array([0,0,0]), world_x_axis)
                            log_vector(rr, "world_y_axis", np.array([0,0,0]), world_y_axis)
                            log_vector(rr, "world_z_axis", np.array([0,0,0]), world_z_axis)
                        
                    # object is moving with the user so do not iclude it
                    if object_name in objects_in_motion:
                        continue
                    
                    # store the object names that satisfy the criteria for distance 
                    objects_names_low_distance_values[object_name] = f"{float(filtered_distances[index]):.3f}"                                        # Me: keep only one value from the floating form
                    objects_names_low_distance_counts[object_name] = filtered_names_len_low_distance_counts[object_name]
                    
                    # store the list and dictionary with object namea and time
                    objects_time_approach_list.append(object_time_xz)                                                                                  
                    objects_time_approach_dict[object_name] = object_time_xz
                    
                    # store the list and dictionary with object namea and time if and only if the time is less than a threshold
                    if object_time_xz < parameters["time_threshold"]:
                        objects_names_less_than_2_seconds_dict[object_name] = object_time_xz
                        objects_names_less_than_2_seconds_list.append(object_time_xz)   
                        print(f"Time to approach {object_name} with {object_id} is {object_time_xz}")

                    # store the list and dictionary with object name and time where an object satisfy the criteria of time and counter
                    """ 
                    Time is less than 2 seconds + counters. 
                    We store object names which are approachable in less than 2 secomds for a number of counters where - Time Threshold * 30 
                    This shows the user is around these objects but do not have the intention to interact with
                    """ 
                    if object_time_xz < parameters["time_threshold"] and object_id in filtered_time_to_approach_counts and len(filtered_time_to_approach_counts[object_id]) > parameters["time_counters_threshold"]:
                        objects_less_than_2_seconds_with_counts_dict[object_name] = object_time_xz
                        objects_less_than_2_seconds_with_counts_list.append(object_time_xz)  
                    
            ## Identify objects with high duration 
            """
            Loop through the values of the filtered duration 
            """
            for obj_id, values in filtered_duration.items():
                if values[-1][1] > 2:
                    object_name = gt_provider.get_instance_info_by_id(obj_id).name
                    if object_name in objects_in_motion:
                        continue
                    objects_high_duration[object_name] = f"{float(values[-1][1]):.3f}"
                    
            # ==============================================
            # LLM Query and Activation
            # ==============================================  
            
            """
            1. we activate LLM only if there is object that satisfy the criteria of high dot thrshold
            2. we activate LLM only if there is object that satisfy the criteria of the distance threshold
            3. we activate LLM only if there is object that is approachable in less than the time threshold
            
            TODO: update these criteria 
            
            4. we activate LLM only if there is object that has high visibility 
            5. we activate LLM only if there is no object that is approachable in less that 2 seconds but this remains consistent for 2 seconds (means is around this object)
            """
            
            if (objects_names_high_dot_counts
                and objects_names_low_distance_counts
                and objects_names_less_than_2_seconds_dict
                and objects_high_duration
                and not objects_less_than_2_seconds_with_counts_dict
                ):
        
                # Print statement 
                print("the 4th criteria have been satisfied")

                # Write information only if LLM is ON and is ready to activated 
                if args.use_llm and not llm_activated:  
                    
                    # Additional condition to activate the LLM. The object that is approachable in less than 2 seconds should belong in the closed list
                    if any(object in objects_with_high_dot_history for object in objects_names_less_than_2_seconds_dict.keys()): # TODO: make this soft having it as a group of objects that was looking, Red Clock was within a group 
                        
                        """
                        1. objects names --- high dot counts                                name:  high_focus_objects_measured_in_counts
                        2. objects names --- distance counts                                name:  nearby_objects_measured_in_counts
                        3. objects names --- high dot value before activation               name:  objects_names_and_latest_focus_intensity 
                        4. objects names --- distance value before actication               name:  objects_names_and_latest_distance_from_the_user
                        5. objects names --- list of names with time less than 2 seconds    name:  quick_access_object 
                        """
                        
                        # write the log
                        history_log = append_to_history_string(current_time_s, 
                                                "Living Room", 
                                                objects_names_high_dot_counts,
                                                objects_names_low_distance_counts,  
                                                objects_names_low_distance_values,
                                                objects_names_less_than_2_seconds_list,      # filtered_names_len_time_counts, 
                                                predictions_dict,
                        )
            
                        # Convert history log to a string
                        history_log_string = str(history_log)
                        
                        # use the LLM
                        llm_response = activate_llm(history_log_string)
                        
                        # write the lllm 
                        llm_path = "llm/llm_response.yaml"
                        llm_folder = os.path.join(project_path, llm_path)   
                        os.makedirs(os.path.dirname(llm_folder), exist_ok=True)   # check existence 
                        with open(llm_folder, "a") as file:
                            file.write("\n\n")
                            file.write(llm_response)
                        print(f"LLM response has been written to {llm_folder}")

                        # Update the last time LLM was activated 
                        last_activation_time = current_time_s
                        
                        if True: # if llm_response:
                            
                            predicted_objects, goal = process_llm_response(llm_response)
                            llm_activated = True
                            user_relative_total_movement = 0

                            # Keep the output of LLM in log file   # TODO: update the log file for 1 second
                            log_filename = f'logs/time_{current_time_s}.log'    
                            log_folder = os.path.join(project_path, log_filename)               
                            os.makedirs(os.path.dirname(log_folder), exist_ok=True)
                            logger = setup_logger(log_folder)
                            logger.info(f"LLM Response: {llm_response}")

                            # append the object predictios and goal to a list 
                            predictions_dict[current_time_s] = predicted_objects
                            goals_dict[current_time_s] = goal
                            predictions.append(predicted_objects)
                            llm_times.append(current_time_s)
                            
                            # Log history_log content until the LLM activation and then initialize it again 
                            history_log_filename = f'logs/history_{current_time_s}.log'
                            history_log_folder = os.path.join(project_path, history_log_filename)
                            os.makedirs(os.path.dirname(history_log_folder), exist_ok=True)
                            history_logger = setup_logger(history_log_folder)
                            history_logger.info(f"History Log: {history_log}")

                            write_to_excel(filtered_names_len_high_dot_counts, filtered_names_len_low_distance_counts, filtered_object_time_interaction, filtered_names_dot, filtered_names_distances, predictions_dict, goals_dict, args.sequence_path, parameter_folder_name, current_time_s)
            
            # ==============================================
            # Objects Visualization
            # ==============================================  
            
            # if args.runrr and args.visualize_objects:
            
            #     for obj_id, dot_product, distance, obj_position_cam, obj_position_scene in zip(
            #                                                                             filtered_obj_ids, 
            #                                                                             filtered_dot_products,
            #                                                                             filtered_distances,
            #                                                                             filtered_obj_positions_cam, 
            #                                                                             filtered_obj_positions_scene,
            #                                                                             # objects_time_approach_list,
            #                                                                             ):

            #         # instance info 
            #         instance_info = gt_provider.get_instance_info_by_id(obj_id)
                    
            #         # name
            #         object_name = instance_info.name
                    
            #         # Handling the object coordinates 
            #         """
            #         ** bbox_3d.aabb --> represents the minimum and maximum coordinates that define the AABB --> [x_min, y_min, z_min, x_max, y_max, z_max]
            #         ** bbox_3d.transform_scene_object is a transformation from οbject coordinate system to scene coordinate system (4x4 matrix)
            #         ** AABB - Axis-Aligned Bounding Box ---> 16 points that form 8 pairs (edges), which define a 3D bounding box. Some of these points are repeated.
            #         ** OBB  - Oriented bounding box     ---> same dimension with aabb_coords. it will store the values on the new coordinate system
            #         """
            #         bbox_3d = bboxes3d[obj_id]                                                     
            #         aabb_coords = bbox3d_to_line_coordinates(bbox_3d.aabb)                         
            #         obb = np.zeros(shape=(len(aabb_coords), 3))                                    
            #         for i in range(0, len(aabb_coords)):                                           
            #             aabb_pt = aabb_coords[i]                                                  
            #             aabb_pt_homo = np.append(aabb_pt, [1])                                    
            #             obb_pt = (bbox_3d.transform_scene_object.to_matrix() @ aabb_pt_homo)[0:3]  
            #             obb[i] = obb_pt

            #         # Prints # 
            #         # print(f"Object name: {instance_info.name}, Dot Product: {dot_product}, Distance: {distance}, Position on scene {bbox_3d.transform_scene_object.translation()}")                
            #         # print(f"Position on Camera frame: {obj_position_cam}, Position on Scene frame: {obj_position_scene}")
            #         # print("-" * 40)  # Print a line of dashes as a separator | # More sophisticated separator)
                        
            #         # ==============================================
            #         # VECTORS - POSITION / VELOCITY / DISTANCE / PROJECTION  
            #         # ==============================================  
                    
            #         # POSITION - vectors
            #         position_vector_xyz = T_Scene_Device.translation()[0]   
            #         position_vector_xz = np.array([position_vector_xyz[0], 0, position_vector_xyz[2]])
                    
            #         # VELOCITY - vector in 3D
            #         velocity_vector_xyz = T_Scene_Device.rotation().to_matrix() @ user_velocity_device
            #         velocity_vector_xz = np.array([velocity_vector_xyz[0], 0, velocity_vector_xyz[2]])
            #         velocity_vector_xyz_on_user = velocity_vector_xyz + position_vector_xyz     # End of the velocity vector (scaled by velocity)
            #         velocity_vector_xz_on_user = velocity_vector_xz + position_vector_xyz       # Velocity vector in 2D 
                    
            #         # DISTANCE vector in 3D from user to object
            #         displacement_vector = (obj_position_scene[0] - position_vector_xyz) 
            #         displacement_unit_vector = displacement_vector / np.linalg.norm(displacement_vector)
            #         displacement_vector_xz = np.array([displacement_vector[0], 0, displacement_vector[2]])
            #         displacement_unit_vector_xz = np.array([displacement_unit_vector[0], 0, displacement_unit_vector[2]])
                    
            #         # DISTANCE vector in 3D from user to object ON USER
            #         displacement_vector_on_user = (obj_position_scene[0] - position_vector_xyz) + position_vector_xyz
            #         displacement_unit_vector_on_user = displacement_vector / np.linalg.norm(displacement_vector) + position_vector_xyz
            #         displacement_vector_xz_on_user = np.array([displacement_vector[0], 0, displacement_vector[2]]) + position_vector_xyz
            #         displacement_unit_vector_xz_on_user = np.array([displacement_unit_vector[0], 0, displacement_unit_vector[2]]) + position_vector_xyz
                    
            #         # PROJECTED VELOCITY vector in 3D from user to object  (Project the user's velocity onto the displacement vector)
            #         projected_velocity_xyz = np.dot(velocity_vector_xyz, displacement_unit_vector) * displacement_unit_vector 
            #         projected_velocity_xz = np.array([projected_velocity_xyz[0], 0, projected_velocity_xyz[2]])
            #         projected_velocity_xyz_on_user = projected_velocity_xyz + position_vector_xyz 
            #         projected_velocity_xz_on_user = projected_velocity_xz + position_vector_xyz
                   
            #         # ==============================================
            #         # VISUALISATION ORIGIN
            #         # ==============================================  
                    
            #         # # POSITION - Visualize the position vector (3D & 2D)
            #         # args.runrr and log_vector(rr, "origin_position_xyz", np.array([0,0,0]), position_vector_xyz)
            #         # args.runrr and log_vector(rr, "origin_position_ema_xyz", np.array([0,0,0]), user_ema_position)
            #         # args.runrr and log_vector(rr, "origin_position_xz", np.array([0,0,0]), position_vector_xz)
                    
            #         # # VELOCITY - Visualize the velocity vector as a line in the GUI
            #         # args.runrr and log_vector(rr, "origin_velocity_xyz", np.array([0,0,0]), velocity_vector_xyz)
            #         # args.runrr and log_vector(rr, "origin_velocity_ema_xyz", np.array([0,0,0]), user_ema_velocity)
            #         # args.runrr and log_vector(rr, "origin_velocity_xz", np.array([0,0,0]), velocity_vector_xz)
                    
            #         # # PROJECTED VELOCITY - Visualize the velocity vector projected towards the  
            #         # args.runrr and log_vector(rr, "origin_projected_velocity_xyz", np.array([0,0,0]), projected_velocity_xyz)
            #         # args.runrr and log_vector(rr, "origin_projected_velocity_xz", np.array([0,0,0]), projected_velocity_xz)
                    
            #         # # DISTANCE 3D - Visualize the position vector (3D & 2D)
            #         # args.runrr and log_vector(rr, "origin_distance_vector_xyz", np.array([0,0,0]), displacement_vector)
            #         # args.runrr and log_vector(rr, "origin_distance_unit_vector_xyz", np.array([0,0,0]), displacement_unit_vector)
            #         # args.runrr and log_vector(rr, "origin_distance_vector_xz", np.array([0,0,0]), displacement_vector_xz)
            #         # args.runrr and log_vector(rr, "origin_distance_unit_vector_xz", np.array([0,0,0]), displacement_unit_vector_xz)
                    
            #         # # OBJECT POSITION VECTOR - Visualise the position of the vector in the space 
            #         # args.runrr and log_vector(rr, "origin_object_position", np.array([0,0,0]), obj_position_scene[0])
            #         # args.runrr and log_vector(rr, "camera_to_object", camera_position_scene, filtered_vector_camera_objects_scene[0] + camera_position_scene)
            #         # args.runrr and log_vector(rr, "origin_camera_to_object", np.array([0,0,0]), filtered_vector_camera_objects_scene[0])
                    
            #         # ==============================================
            #         # VISUALISATION ON USER 
            #         # ==============================================  
                    
            #         # # VELOCITY - Visualize the velocity vector as a line in the GUI
            #         args.runrr and log_vector(rr, "device_velocity_xyz", position_vector_xyz, velocity_vector_xyz_on_user)
            #         # args.runrr and log_vector(rr, "device_velocity_xz", position_vector_xyz, velocity_vector_xz_on_user)
                    
            #         # # PROJECTED VELOCITY - Visualize the velocity vector projected towards the  
            #         # args.runrr and log_vector(rr, f"device_projected_velocity_xyz", position_vector_xyz, projected_velocity_xyz_on_user)
            #         # args.runrr and log_vector(rr, f"device_projected_velocity_xz", position_vector_xyz, projected_velocity_xz_on_user)
                    
            #         # # DISTANCE 3D - Visualize the position vector (3D & 2D) - ON USER 
            #         # args.runrr and log_vector(rr, f"device_distance_vector_xy", position_vector_xyz, displacement_vector_on_user)
            #         # args.runrr and log_vector(rr, f"device_distance_unit_vector_xyz", position_vector_xyz, displacement_unit_vector_on_user)
            #         # args.runrr and log_vector(rr, f"device_distance_vector_xz", position_vector_xyz, displacement_vector_xz_on_user)
            #         # args.runrr and log_vector(rr, f"device_distance_unit_vector_xz", position_vector_xyz, displacement_unit_vector_xz_on_user)
                    
            #         # OBJECT LINE - Visualize the line from camera to object
            #         # args.runrr and log_object_line(rr, instance_info, position_vector_xyz, obj_position_scene[0])
            #         # args.runrr and log_object_line(rr, instance_info, camera_position_scene[0], obj_position_scene[0])

            #         # OBJECT BOUNDING BOX - Visualize the object
            #         args.runrr and log_object(rr, instance_info, obb)
                    
            #         # ==============================================
            #         # VISUALISATION FRAMES
            #         # ==============================================    
                    
            #         # # CAMERA Z-AXIS ONLY ROTATION TO SCENE
            #         # cam_z_axis_rotation = T_Scene_Cam.rotation().to_matrix() @ cam_z_axis     
            #         # args.runrr and log_vector(rr, "origin_camera_z_axis_rotation_only", np.array([0,0,0]), cam_z_axis_rotation)
            #         # args.runrr and log_vector(rr, "origin_camera_z_axis", np.array([0,0,0]), cam_z_axis_scene)
                    
            #         # # CAMERA
            #         # args.runrr and log_vector(rr, "camera_x_axis", camera_position_scene[0], cam_x_axis_scene)
            #         # args.runrr and log_vector(rr, "camera_y_axis", camera_position_scene[0], cam_y_axis_scene)
            #         # args.runrr and log_vector(rr, "camera_z_axis", camera_position_scene[0], cam_z_axis_scene)
                    
            #         # # DEVICE
            #         # args.runrr and log_vector(rr, "device_x_axis", position_vector_xyz, device_x_axis_scene)
            #         # args.runrr and log_vector(rr, "device_y_axis", position_vector_xyz, device_y_axis_scene)
            #         # args.runrr and log_vector(rr, "device_z_axis", position_vector_xyz, device_z_axis_scene)
                    
            #         # # WORLD
            #         # args.runrr and log_vector(rr, "world_x_axis", np.array([0,0,0]), world_x_axis)
            #         # args.runrr and log_vector(rr, "world_y_axis", np.array([0,0,0]), world_y_axis)
            #         # args.runrr and log_vector(rr, "world_z_axis", np.array([0,0,0]), world_z_axis)
                    
            #         # Add to previous_obj_ids to clear in the next timestep
            #         previous_obj_names.add(instance_info.name)
            #         previous_obj_ids.add(obj_id)

            #         if gt_provider.get_instance_info_by_id(object_id).name == "ChoppingBoard":
            #             print('stop')
                        
            # ==============================================
            # Objects Inside the radius & LLM activation conditions
            # ==============================================  
            
            # Get objects within 1.5 meter radius 
            current_objects_within_radius = object_within_radius(visible_distance_device_objects_scene, visible_obj_names, radius = 1.5)
            
            # Calculate conditions
            group_analyzer.add_objects(current_time_s, current_objects_within_radius)
            user_objects = group_analyzer.compare_objects()                                   # Me: Boolean value (True/False) if user moved to different area based on objects around
            users_move = group_analyzer.user_move(user_relative_total_movement)               # Me: Boolean value (True/False) if user moved to different area based on movement
            time_since_last_activation = current_time_s - last_activation_time                # Me: Time 
        
            # Conditions to enable the LLM
            """
            4 conditions for the LLM
            - Not earlier than 2 seconds 
            - Not later than 5 seconds 
            - user's movement significant 
            - user's is not surrounding by the same of objects 
            """
            
            if time_since_last_activation > 2: 
                if users_move or user_objects or time_since_last_activation > 5:
                    user_relative_total_movement = 0
                    last_activation_time = current_time_s
                    llm_activated = False
        
        # ==============================================
        # Store the predictions
        # ==============================================  
        
        # Define the path for saving the predictions
        predictions_folder = os.path.join(project_path, 'data', 'predictions', sequence_path, parameter_folder_name)
        os.makedirs(predictions_folder, exist_ok=True)

        # Save the predictions to a JSON file
        prediction_file = os.path.join(predictions_folder, 'large_language_model_prediction.json')
        with open(prediction_file, 'w') as json_file:
            json.dump(predictions_dict, json_file, indent=4)

        print(f"Saved predictions for parameters to {prediction_file}")
        
        # ==============================================
        # Prints
        # ==============================================  
        
        # Print basic sequence characteristics
        print(f"Loaded scene: {dataset_folder}")
        print("Scene characteristics:")
        print(f"\t Aria RGB frames count: {len(img_timestamps_ns)}")
        print(f"\t Skeleton count: {len(gt_provider.get_skeleton_ids())}")
        
        # Print objects info
        # print(f"\t Static object count: {len(static_obj_ids)}")
        # print(f"\t Dynamic object count (tracked): {len(dynamic_obj_pose_cache)}")
        # print(f"\t Dynamic object count (tracked and moved - estimated): {len(dynamic_obj_moved)}")

    if __name__ == "__main__":
        main()


