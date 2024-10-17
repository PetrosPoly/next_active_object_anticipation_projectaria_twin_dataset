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
    AriaGlassesOutline,                                        # Me: Return a list of points to be used to draw the outline of the glasses (line strip).
    ToTransform3D                                              # Me: Helper function to convert Sophus SE3D pose to a Rerun Transform3D
)                                                               

from visualization.rr import (                                # Me: added by Petros
    initialize_rerun_viewer,                                
    log_camera_calibration,
    log_aria_glasses,
    set_rerun_time,
    process_and_log_image,
    log_device_transformations,
    log_dynamic_object,
    log_object,
    log_object_line,
    clear_logs_names,
    clear_logs_ids,
)

from utils.tools import (
    transform_point,                                          # Me: Transformation point from scene to camera frame
    visibility_mask,                                          # Me: Check which points are visible and which are not visible
    time_to_interaction,                                      # Me: Time to interact with each object
    exponential_filter,                                       # Me: Filter the velocity with exponential mean average 
    object_within_radius,                                     # Me: Check the objects that are close to a user
    user_movement_calculation,                                # Me: User's movement calculation
    calculate_relative_pose_difference                        
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
    Statistics                                          # Me: Keep statistics for high dot value and low distance
)

from utils.evaluation import (
    LLMEvaluation
)

from utils.globals import (
    HISTORY_WINDOW_TIME,       
    VARIABLES_WINDOW_TIME,     
    HIGH_DOT_THRESHOLD,  
    DISTANCE_THRESHOLD,     
    LOW_DISTANCE_THRESHOLD,
    TIME_THRESHOLD,        
    HIGH_DOT_COUNTERS_THRESHOLD,      
    LOW_DISTANCE_COUNTERS_THRESHOLD,
    TIME_COUNTERS_THRESHOLD,
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

def main():
    # Files 
    args = parse_args()
    
    base_path = ["Apartment_release_work_seq107_M1292",  # work 107
                 "Apartment_release_clean_seq150_M1292", # clean 150
                 ""]                                     # ?   
    
    base_folder = os.path.join(args.sequence_path, base_path[1])
    vrsfile = os.path.join(base_path[1], "video.vrs")
    ADT_trajectory_file = os.path.join(base_path[1], "aria_trajectory.csv")
    
    # Path to log items for the LLM - Define the CSV file to log the items and check if it exists to write the header
    folder = "Documents/projectaria_sandbox/projectaria_tools/projects/AriaDigitalTwinDatasetTools/object_anticipation/adt/"
    csv_file = os.path.join(folder,'interaction_log.csv')

    # Print the paths
    print("Sequence_path: ", base_folder)
    print("base_path", base_path)
    print("VRS File Path: ", vrsfile)  
    print("GT trajectory path: ", ADT_trajectory_file)
    
    try:
        paths_provider = AriaDigitalTwinDataPathsProvider(base_folder)
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
      
    ####                                   #### 
    ####    INITIALIZATION OF VARIABLES    ####
    ####                                   ####
    
    # Store a Pose cache for dynamic object, so we log them only if they changed position
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
    very_close_distance_counts: Dict[int, List[Tuple[int, float]]] = defaultdict(list) 
    
    # List of ground truth objects that the user interacted with 
    ground_truth = ['ChoppingBoard',   
                    '',                                             
                    ['KitchenKnife', 'WoodenSpoon'], 
                    'WoodenFork', 
                    'Donut_B', 
                    'Cereal_Anon', 
                    'WoodenToothbrushHolder',
                    'Cracker_Anon', 
                    'BlackCeramicMug']
    
    # Activation of LLM
    llm_activated = False                                                           # Me: LLM activated or not
    # history_log: Deque[Dict] = deque()                                            # Me: Initialize the history log as a deque. This was needed when we put the old values 
    history_log: Dict                                                               # Me: This is for now 
    objects_high_dot_history = []                                                   # Me: initialize a history high dot objects  /  scope of this is to check if the object that satisy my criteria for time once belongs to the high dot
    predictions = []                                                                # Me: Predictions of the LLM      
    predictions_dict = {}                                                           # Μe: Predictions of the LLM as a dictionary with the timestamps
    goals_dict = {}
    goals_list = []
    llm_times = []                                                                  # Me: Timestamps that LLM predicted each output
    user_velocity_before = None                                                     # Me: Initialize previous velocity
    user_position_before = None                                                     # Me: Calculate the user's position
    objects_within_radius = []                                                      # Me: Objects in the vicinity of the user
    previous_objects_within_radius = []                                             # Me: Is used to check if the user is stll in the same area that LLM has been activated in order to avoid reactivatiomn of the LLM 
    last_activation_time = 0                                                        # Me: the time activatiom of an LLM
    all_unique_object_names_with_high_dot = set()
    last_time_LLM_actvated = 0                                                      
    
    # Movement detection phase / we need the old poses of the objects
    detection_phase = True                                                          # Me: Detection phase boolean factor  
    T_scene_object_before = None                                                    # Me: Objects' poses to check a relative transformation between poses for each round
    counter = -1                                                                    # Me: counter of objects
    check_ground_truth = 0.033                                                      # Me: 1 second time interval to check for movement
    last_detection_checked_time = 0                                                 # Me: last time we checked for pose change 
    original_objects_that_moved_dict = {}                                           # Me: Dict with the objects that user interacted with (dictionary)
    movement_time_dict = {}
    original_objects_that_moved_list = []                                           # Me: List with the objects that user interacted with (list)
    result_objects_that_moved_dict = {}  
    indexes_of_objects_that_moved = []                                              # Me: List with the indexes of the objects that moved (indexes from the general list of objects generated with the bboxes list
    previous_moved_names = []                                                       # Me: List of objects that previously moved
    previous_name = []                                                              # Me: Initialize previous_name
    objects_the_user_is_moving_with = {}
    distances = {}
    
    # Check if users has an object with him and is moving 
    user_object_position = {}                                                       # User's position when a specific object is inside a radius of 1.5 meters 
    user_object_movement = {}                                                       # User's distance from initial position
    user_object_movement_abs = {}                                                   # User's distance from last position. Increment
    
    # For debugging 
    value_pose = {}                                                                 # Me: DEBUGGING: Pose of the values                                 
    means = {}                                                                      # Me: DEBUGGING: mean value to keep the statistics.
    st_dev = {}                                                                     # Me: DEBUGGING: standard deviation to keep the std. 
    times = []                                                                      # Me: DEBUGGING: For debugging again to keep the values.
    include_cutting_object = True                                                   # Me: include the main cutting board object
    objects_number_with_radius = []                                                 # Me: a list with the number 
    
    # Initialize classes
    group_analyzer = ObjectGroupAnalyzer(history_size=15, future_size=5, change_threshold=0.7)                                          # Me: Group check
    statistics = Statistics(VARIABLES_WINDOW_TIME, HIGH_DOT_THRESHOLD, DISTANCE_THRESHOLD, LOW_DISTANCE_THRESHOLD, TIME_THRESHOLD)      # Me: Initialize the ObjectStatistics instance
    
    with open(os.path.join(folder,'movement_time_dict.json', 'r')) as json_file:
        movement_time_dict = json.load(json_file)
    
    with open(os.path.join(folder,'objects_that_moved.json', 'r')) as json_file:
        original_objects_that_moved_dict = json.load(json_file)
    
    # For loop from the start of the time till the end of it
    for timestamp_ns in tqdm(img_timestamps_ns):
        args.runrr  and set_rerun_time(rr, timestamp_ns)
       
        ## Print the current time
        current_time_ns = timestamp_ns
        current_time_s = round((current_time_ns / 1e9 - start_time), 3)
        
        ## Clear previously logged objects and lines
        if args.runrr:
            clear_logs_ids(rr, previous_obj_ids)
            clear_logs_names(rr, previous_obj_names)
        
        previous_obj_ids.clear()
        previous_obj_names.clear()
        
        ## Log RGB image
        image_with_dt = gt_provider.get_aria_image_by_timestamp_ns(timestamp_ns, rgb_stream_id)
        args.runrr and process_and_log_image(rr, args, image_with_dt)
        
        ####                                                      #### 
        ####        USERS POSES - POSITION/MOVEMENT (SCENE)       ####
        ####                                                      ####
                                                                              
        ## Device pose in scene frame                                                                          
        aria_3d_pose_with_dt = gt_provider.get_aria_3d_pose_by_timestamp_ns(timestamp_ns)                                 # Me: Pose of the device

        if aria_3d_pose_with_dt.is_valid():
            p_dev_scene = aria_3d_pose_with_dt.data()                                                                     # Me: Pose of the user's device on Scene frame
            T_Scene_Device = p_dev_scene.transform_scene_device                                                           # Me: SE3 of the device 
            T_Scene_Cam = T_Scene_Device @ T_Device_Cam                                                                   # Me: SE3 from Camera to Scene
            
            ## User's position and velocity in the scene frame 
            user_position_scene = aria_3d_pose_with_dt.data().transform_scene_device.translation()[0]      
            user_velocity_scene = aria_3d_pose_with_dt.data().device_linear_velocity
            
            ## User filtered position - velocity 
            if user_position_before is None and user_velocity_before is None:
                
                # user's position & velocity 
                user_ema_position = user_position_before = user_position_scene  
                user_ema_velocity = user_velocity_before = user_velocity_scene   
                
                # Initialise user's movement 
                user_total_movement = 0                                                                                   
                user_relative_total_movement = 0                                                                          # Me: This is the movement of the user every two
                
                # user's position store 
                users_position = deque([(current_time_s, 0)])           
                users_velocity = deque([(current_time_s, 0)])     
                
            else:
                # user's position and velocity
                user_ema_position = exponential_filter(user_position_scene, user_position_before, alpha = 0.9)            # Me: Exponential filter for the position to reduce the noise 
                user_ema_velocity = exponential_filter(user_velocity_scene, user_velocity_before, alpha = 0.9)            # Me: Apply exponential filter 
                
                # user's movement at each timestamp 
                user_movement_timestep = user_movement_calculation(user_position_before, user_ema_position) 
                user_total_movement += user_movement_timestep
                user_relative_total_movement += user_movement_timestep
                
                # user's position store
                users_position.append((current_time_s, user_ema_position[:2]))                                            # Me: collect user's position on x,y 
                users_velocity.append((current_time_s, user_ema_velocity[:2]))       
                
                # recalculate the position & velocity before
                user_position_before = user_ema_position
                user_velocity_before = user_ema_velocity
                
            ## Insert this info in the runio.rr software
            args.runrr and log_device_transformations(rr, p_dev_scene, T_Device_Cam, ToTransform3D) 
        
        # print('user total movement is the following', user_total_movement)
        
        ####                                      #### 
        ####            OBJECTS POSES             ####
        ####                                      ####
        
        ## Objects informations (orientation) for each timestamp in nanoseconds 
        bbox3d_with_dt = gt_provider.get_object_3d_boundingboxes_by_timestamp_ns(timestamp_ns)
        assert bbox3d_with_dt.is_valid(), "3D bounding box is not available"
        bboxes3d = bbox3d_with_dt.data()                                                                                  # Me: Objects data
                                                                                                                    
        ## Extract object IDs and their positions
        obj_ids = np.array(list(bboxes3d.keys()))                                                                         # Me: Ids of the objects
        obj_number = len(obj_ids)
        obj_names = np.array([gt_provider.get_instance_info_by_id(obj_id).name for obj_id in obj_ids])                    # Me: Names of the objects
        obj_positions_scene = np.array([bbox_3d.transform_scene_object.translation() for bbox_3d in bboxes3d.values()])   # Me: Positions on Scene frame #TODO: maybe add the [0] to take the first elemeent 
        distance_user_objects_scene = np.linalg.norm(obj_positions_scene.reshape(obj_number,3) - user_position_scene, axis=1)    # Me: Distances from user's device to objects on scene frame   
        
        ####                                       #### 
        ####    VISIBLE OBJECTS IN CAMERA FRAME    ####
        ####                                       ####
        
        ## Objects in Camera Frame
        T_Cam_Scene = T_Scene_Cam.inverse()                                                                               # Me: Transform positions to the camera frame
        obj_positions_cam = np.array([transform_point(T_Cam_Scene, pos.reshape(1, 3)) for pos in obj_positions_scene])    # Me: Positions on Camera frame
        obj_positions_cam_reshaped = obj_positions_cam.reshape(-1, 3, 1)                                                  # Me: Prepare positions for projection
        
        ## Produce the visible mask to work only with objects that are in the camera frame
        valid_mask = visibility_mask(obj_positions_cam_reshaped, rgb_camera_calibration)                                  # Me: Produce the Mask
        
        ## Take the only poses of the visible objects   
        T_scene_object = {}
        for key, (include, value) in zip(bboxes3d.keys(), zip(valid_mask, bboxes3d.values())):                            # Me: Filter the dictionary based on the boolean array
            if True: # if include:
                T_scene_object[key] = value.transform_scene_object
        
        ## Take ids, names, position and distances for the visible objects 
        visible_obj_ids = obj_ids[valid_mask]                                                                             # Me: Filter the ids of visible objects
        visible_obj_names = obj_names[valid_mask]                                                                         # Me: Filter the names of visible objects
        visible_obj_positions_scene = obj_positions_scene[valid_mask]                                                     # Me; Filter the positions visible objects in scene 
        visible_obj_positions_cam = obj_positions_cam[valid_mask]                                                         # Me: Filter the positions visible objects in camera
        visible_distance_user_objects_scene = distance_user_objects_scene[valid_mask]                                     # Me: Filter the distances from device to visible objects in scene
        
        ####                                             #### 
        ####     DOT PRODUCTS DISTANCES & VELOCITIES     ####
        ####                                             ####
        
        ## Dot Products and Line Distances from Camera to Objects in scene frame
        cam_z_axis = np.array([0, 0, 1]).reshape(3, 1)
        cam_z_axis_scene = T_Scene_Cam.rotation() @ cam_z_axis                                                            # Me: Camera Z-axis on Scene frame
        p_cam_scene = T_Scene_Cam.translation()                                                                           # Me: Camera position in scene
        vector_cam_to_objs = visible_obj_positions_scene[:, 0] - p_cam_scene                                              # Me: Vector from camera to objects in scene
        vector_cam_to_objs_normalized = vector_cam_to_objs / np.linalg.norm(vector_cam_to_objs, axis=1, keepdims=True)    # Me: Normalize the vectors
        dot_products = np.dot(vector_cam_to_objs_normalized, cam_z_axis_scene.squeeze())                                  # Me: Dot Products Calculation
        distance_camera_objects_scene = np.linalg.norm(vector_cam_to_objs, axis=1)                                        # Me: Distance Calculation from camera to objects 

        ## Calculate the advanced criteria
        dot_products_list = dot_products.tolist()                                                                         # Me: Calculate the dot products 
        distance_camera_objects_scene_list = distance_camera_objects_scene.tolist()                                       # Me: Calculate the line distances
        
        ## Time Difference
        time_diff_ns = (current_time_ns - previous_time_ns) / 1e9                                                         # Me: Calculate the time difference in seconds
        previous_time_ns = current_time_ns

        ####                                                  #### 
        ####        OBJECT INTERACTION DETECTION PHASE        ####
        ####                                                  ####
            
        # Get the L2 norm of the angle-axis vector and translation vector (after the log operation in the SE3 for the relative transformation)
        norm_relative_T_object, T_scene_object_before = calculate_relative_pose_difference(T_scene_object_before, T_scene_object)

        # Find the object indexes that satisfy your condition for potential interaction 
        indexes_activation = np.where((norm_relative_T_object) > 0.004)[0] # 4mm 
        objects_that_is_moving = list(obj_names[indexes_activation])
        
        # If Condition to check interaction took place (apart from norm of relative transformation need to check also the distance which should be relative close)
        if indexes_activation.size > 0: 
            
            # use the indexes to find THE potential objects names that moved and respective distance from them:
            objects_that_moved_names = list(obj_names[indexes_activation])
            
            for i, (name, index) in enumerate(zip(objects_that_moved_names, indexes_activation)):
                
                """
                1. loop over the objects that moved based on pose change and the list of respective indexes calculated from the object names list 
                2. store the distance of the user from the object
                3. store the position of the user and the movement of the user for the object that has been moved 
                4. check if in the previous step the object has been moved in order to avoid insert it in the dictionary original_objects_that_moved_dict  
                """
                
                if name not in distances:
                    distances[name] = [distance_user_objects_scene[index]]
                    user_object_position[name] = [user_ema_position[:2]]
                    user_object_movement[name] = 0
                else:
                    distances[name].append(distance_user_objects_scene[index])
                    user_object_position[name].append(user_ema_position[:2]) # only xy plane
                    user_object_movement[name] += np.linalg.norm(user_object_position[name][-1] - user_object_position[name][-2]) 
        
                # fill the ground truth 
                if name not in previous_moved_names:
                    
                    original_objects_that_moved_dict[current_time_s] = name
                    previous_moved_names = objects_that_moved_names 
                    print("Objects that have been moved so far:", original_objects_that_moved_dict)
                    
                    indexes_of_objects_that_moved.append(indexes_activation[i])
                    counter += 1

                # Track the start and end times of the object movement
                if name not in movement_time_dict:
                    # Object starts moving, record the start time
                    movement_time_dict[name] = {"start_time": current_time_s, "end_time": None}
                else:
                    # Object is already moving, update the end time
                    movement_time_dict[name]["end_time"] = current_time_s

    
       
        ####                        #### 
        ####       TIME WINDOW      ####
        ####                        ####
        
        # Update statistics   
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
         avg_dots_list, avg_distances_list) = statistics.time_window(
                                                            current_time_s, 
                                                            time_diff_ns, 
                                                            user_ema_position, 
                                                            user_ema_velocity,
                                                            visible_obj_positions_scene,
                                                            visible_obj_ids, 
                                                            dot_products_list, 
                                                            visible_distance_user_objects_scene,
                                                            # line_distances_cam_list, 
                                                            )

        ####                         #### 
        ####    FILTERING PROCESS    ####
        ####                         ####
        
        # Use cumulative_dot_products for filtering logic
        if average == True:
            
            # high dot mask and relaxed distance
            high_dot_mask = np.array([avg_dots[obj_id] > 0.7 for obj_id in visible_obj_ids])                      # Me: high dot mask shape for those objects that have high accummulated dot
            high_distance_mask = np.array([avg_distances[obj_id] < 3 for obj_id in visible_obj_ids])              # Me: high distance mask
            
            # low average dot mask but also low distance
            low_dot_mask = np.array([avg_distances[obj_id] > 0.2 for obj_id in visible_obj_ids])                  # Me: low dot mask
            low_distance_mask = np.array([avg_distances[obj_id] < 1 for obj_id in visible_obj_ids])               # Me: low distance mask (because the minimum distace is 0.56 from the object)
            
            # combined masks
            combined_high_high_mask = high_dot_mask & high_distance_mask
            combined_low_low_mask = low_dot_mask & low_distance_mask
            
            # combimed low mask
            combined_mask = combined_high_high_mask | combined_low_low_mask
        else:                                                                                                     # Me: the Simple
            high_dot_mask = dot_products > 0.8                                                                    # Me: high dot mask shape is shape (189,) 1D array
            high_distance_mask = distance_camera_objects_scene_list < 3                                           # Me: high distance mask
            low_dot_mask = dot_products > 0.2                                                                     # Me: low dot mask 
            low_distance_mask = distance_camera_objects_scene_list < 0.9                                          # Me: low distance mask (because the minimum distace is 0.56 from the object)
            combined_high_high_mask = high_dot_mask & high_distance_mask                                          # Me: if the distance is high the dot should be high to accept 
            combined_low_low_mask = low_dot_mask & low_distance_mask                                              # Me: if the dot is low the distance should be low to accept 
            combined_mask = combined_high_high_mask | combined_low_low_mask

        # Filter IDs, positios, dot_producs and distances based on the combined mask
        filtered_obj_ids = visible_obj_ids[combined_mask]
        filtered_obj_positions_scene = visible_obj_positions_scene[combined_mask]
        filtered_obj_positions_cam = visible_obj_positions_cam[combined_mask]
        filtered_dot_products = dot_products[combined_mask]
        filtered_line_distances = distance_camera_objects_scene[combined_mask]
   
        ## Filter the visibility_counter and visibility_duration based on the combined_mask                      
        filtered_counter = {obj_id: visibility_counter[obj_id] for obj_id in filtered_obj_ids if obj_id in visibility_counter}     
        filtered_duration = {obj_id: visibility_duration[obj_id] for obj_id in filtered_obj_ids if obj_id in visibility_duration}
        filtered_high_dot_counts = {obj_id: high_dot_counts[obj_id] for obj_id in filtered_obj_ids if obj_id in high_dot_counts}
        filtered_low_distance_counts = {obj_id: close_distance_counts[obj_id] for obj_id in filtered_obj_ids if obj_id in close_distance_counts}
        filtered_time_to_approach_counts = {obj_id: time_to_approach_counts[obj_id] for obj_id in filtered_obj_ids if obj_id in time_to_approach_counts}
        
        ## Dictionaries with object name and number of counts which satisfy the special criteria for the high dot product / distance line and the time duration
        filtered_names_len_high_dot_counts = {gt_provider.get_instance_info_by_id(obj_id).name: len(high_dot_counts[obj_id]) for obj_id in filtered_obj_ids if obj_id in high_dot_counts}
        filtered_names_len_low_distance_counts = {gt_provider.get_instance_info_by_id(obj_id).name: len(close_distance_counts[obj_id]) for obj_id in filtered_obj_ids if obj_id in close_distance_counts}
        filtered_names_len_time_counts = {gt_provider.get_instance_info_by_id(obj_id).name: len(time_to_approach_counts[obj_id]) for obj_id in filtered_obj_ids if obj_id in time_to_approach_counts}
        
        ## For Debugging
        filtered_past_dots = {gt_provider.get_instance_info_by_id(obj_id).name: past_dots[obj_id] for obj_id in filtered_obj_ids}
        filtered_past_distances = {gt_provider.get_instance_info_by_id(obj_id).name: past_distances[obj_id] for obj_id in filtered_obj_ids}
        filtered_names_ids = {gt_provider.get_instance_info_by_id(obj_id).name: obj_id for obj_id in filtered_obj_ids} 
        filtered_ids_names = {obj_id: gt_provider.get_instance_info_by_id(obj_id).name for obj_id in filtered_obj_ids}
        filtered_duration_time = {gt_provider.get_instance_info_by_id(obj_id).name: visibility_duration[obj_id][-1] for obj_id in filtered_obj_ids if obj_id in visibility_duration} 
        
        ####                           #### 
        ####    THE FOUR 4 CRITERIA    ####  1. High dot products duration 2. Low distance duration 3. Time to contact 4. High visibility duration
        ####                           ####
        
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
        
        # high dot values and low distance
        objects_names_high_dot_values = {}    
        objects_names_low_distance = {}
               
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

        ## Identify objects with HIGH DOT COUNTS    
        """ 
        1. ---> store the high dot counters <----
        """
        for index, object_id in enumerate(filtered_obj_ids):
            if object_id in filtered_high_dot_counts and len(filtered_high_dot_counts[object_id]) >= HIGH_DOT_COUNTERS_THRESHOLD:               
                object_name = gt_provider.get_instance_info_by_id(object_id).name                                                               
                if object_name in objects_that_is_moving:
                    continue
                
                # store the object names that satisfy the criteria for high dot
                objects_names_high_dot_values[object_name] = f"{float(filtered_dot_products[index]):.3f}"                                        
                
        ## Keep the objects that have shown high dot values until the current time (regardless of the number of counts)
        """
        1. ---> First part put in the history all the keys that have at least one counter with dot value higher than 0.9 <---
        2. ---> Second part has in the history all the keys that have satisfied the criteria of more than 45 at least once <--- I follow this way
        """
        
        # all_unique_object_names_with_high_dot |= set(filtered_names_len_high_dot_counts.keys()) 
        all_unique_object_names_with_high_dot |= set(objects_names_high_dot_values.keys()) 
        objects_with_high_dot_history = list(all_unique_object_names_with_high_dot)     
                                                      
        ## Identify objects with LOW DISTANCE COUNTS
        for index, object_id in enumerate(filtered_obj_ids):
            if object_id in filtered_low_distance_counts and len(filtered_low_distance_counts[object_id]) > LOW_DISTANCE_COUNTERS_THRESHOLD:   # Me: the user is around this object for more than 30 frames within 90 frames
                object_name = gt_provider.get_instance_info_by_id(object_id).name                                                               
                object_time = time_to_interaction(user_ema_position, user_ema_velocity, filtered_obj_positions_scene[index][0])                # Me: time_to_interaction(user's position, user's velocity, object position)
                   
                # object is moving with the user so do not iclude it
                if object_name in movement_time_dict:
                    continue
                
                # store the object names that satisfy the criteria for distance 
                objects_names_low_distance[object_name] = f"{float(filtered_line_distances[index]):.3f}"                                        # Me: keep only one value from the floating form
                
                # store the list and dictionary with object namea and time
                objects_time_approach_list.append(object_time)                                                                                  
                objects_time_approach_dict[object_name] = object_time
                
                # store the list and dictionary with object namea and time if and only if the time is less than a threshold
                if object_time < TIME_THRESHOLD:
                    objects_names_less_than_2_seconds_dict[object_name] = object_time
                    objects_names_less_than_2_seconds_list.append(object_time)   
                    print(f"Time to approach {object_name} with {object_id} is {object_time}")

                # store the list and dictionary with object name and time where an object satisfy the criteria of time and counter
                """ 
                Time is less than 2 seconds + counters. 
                We store object names which are approachable in less than 2 secomds for a number of counters where - Time Threshold * 30 
                This shows the user is around these objects but do not have the intention to interact with
                """ 
                if object_time < TIME_THRESHOLD and object_id in filtered_time_to_approach_counts and len(filtered_time_to_approach_counts[object_id]) > TIME_COUNTERS_THRESHOLD:
                    objects_less_than_2_seconds_with_counts_dict[object_name] = object_time
                    objects_less_than_2_seconds_with_counts_list.append(object_time)  
                
        ## Identify objects with high duration 
        """
        Loop through the values of the filtered duration 
        """
        for obj_id, values in filtered_duration.items():
            if values[-1][1] > 2:
                obj_name = gt_provider.get_instance_info_by_id(obj_id).name
                if object_name in objects_that_is_moving:
                    continue
                objects_high_duration[obj_name] = f"{float(values[-1][1]):.3f}"
                
        ##                                    #### 
        ##            LLM PROMPTING           #### 
        ##                                    ####
        
        """
        1. we activate LLM only if there is object that satisfy the criteria of high dot thrshold
        2. we activate LLM only if there is object that satisfy the criteria of the distance threshold
        3. we activate LLM only if there is object that is approachable in less than the time threshold
        4. we activate LLM only if there is object that has high visibility 
        5. we activate LLM only if there is no object that is approachable in less that 2 seconds but this remains consistent for 2 seconds (means is around this object)
        """
        # if (np.any(np.array(list(filtered_names_len_high_dot_counts.values())) > HIGH_DOT_COUNTERS_THRESHOLD)
        #     and np.any(np.array(list(filtered_names_len_low_distance_counts.values())) > LOW_DISTANCE_COUNTERS_THRESHOLD)
        #     and objects_names_less_than_2_seconds_dict
        #     and objects_high_duration
        #     and not objects_less_than_2_seconds_with_counts_dict                                                            
        #     ):
        
        if (objects_names_high_dot_values
            and objects_names_low_distance
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
                    log the
                    1. objects names --- high dot counts                                name:  high_focus_objects_measured_in_counts
                    2. objects names --- distance counts                                name:  nearby_objects_measured_in_counts
                    3. objects names --- high dot value before activation               name:  objects_names_and_latest_focus_intensity 
                    4. objects names --- distance value before actication               name:  objects_names_and_latest_distance_from_the_user
                    5. objects names --- list of names with time less than 2 seconds    name:  quick_access_object 
                    """
                    
                    # write the log
                    history_log = append_to_history_string(current_time_s, 
                                            "Living Room", 
                                            filtered_names_len_high_dot_counts,
                                            filtered_names_len_low_distance_counts,
                                            objects_names_high_dot_values, 
                                            objects_names_low_distance,
                                            objects_names_less_than_2_seconds_list,      # filtered_names_len_time_counts, 
                                            predictions_dict,
                    )
        
                    # Convert history log to a string
                    history_log_string = str(history_log)
                    
                    # use the LLM
                    llm_response = activate_llm(history_log_string)
                    
                    # write the lllm 
                    llm_path = "llm/llm_response.yaml"
                    llm_folder = os.path.join(folder, llm_path)   
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
                        log_folder = os.path.join(folder, log_filename)               
                        os.makedirs(os.path.dirname(log_folder), exist_ok=True)
                        logger = setup_logger(log_folder)
                        logger.info(f"LLM Response: {llm_response}")

                        # append the predictios to a list 
                        predictions_dict[current_time_s] = predicted_objects
                        goals_dict[current_time_s] = goal
                        predictions.append(predicted_objects)
                        llm_times.append(current_time_s)
                        
                        # Log history_log content until the LLM activation and then initialize it again 
                        history_log_filename = f'logs/history_{current_time_s}.log'
                        history_log_folder = os.path.join(folder, history_log_filename)
                        os.makedirs(os.path.dirname(history_log_folder), exist_ok=True)
                        history_logger = setup_logger(history_log_folder)
                        history_logger.info(f"History Log: {history_log}")

                        # write_to_excel(filtered_names_len_high_dot_counts, filtered_names_len_low_distance_counts, objects_names_high_dot_values, objects_names_low_distance, objects_names_less_than_2_seconds_dict, predictions_dict, goals_dict)

        ####                                                               #### 
        ####            OBJECTS INSIDE THE RADIUS & LLM ACTIVATION         #### 
        ####                                                               ####
        
        # Get objects within 1.5 meter radius 
        current_objects_within_radius = object_within_radius(visible_distance_user_objects_scene, visible_obj_names, 1.5)

        # Add objects to the TimeWindowAnalyzer
        group_analyzer.add_objects(current_time_s, current_objects_within_radius)
        
        # Compare buffers and print the result
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
                
        ####                                       #### 
        ####            LLM ACTIVATION             #### 
        ####                                       ####
         
        # if ((current_time_s - last_time_LLM_actvated) > 2
        #     or user_around_to_new_objects                                   # Me: this combines the user's displacememt and user's
        #     or end_of_interaction):
        
        # if ((current_time_s - last_time_LLM_actvated) > 2
        #     or user_around_to_new_objects):                                   # Me: this combines the user's displacememt and user's
            
        #     # activate the LLM
        #     llm_activated = False 
    
    ####                              #### 
    ####          EVALUATION          #### 
    ####                              ####
    
    # Ground Truth
    # ground_truth = original_objects_that_moved_dict
    filtered_objects = {k: v for k, v in user_object_movement.items() if v > 1.0}
    ground_truth = {k: v for k, v in original_objects_that_moved_dict.items() if v in filtered_objects}  # Create a new dictionary with filtered objects from objects_dict

    # LLM predictions
    LLM_predictions = predictions_dict

    # Initialize the LLMEvaluation class
    evaluation = LLMEvaluation(LLM_predictions, ground_truth)
    
    # Adjust the predictions with ground truth
    correspondances = evaluation.adjust_predictions_with_gt()
    
    # Calculate accuracy
    accuracy, precision, recall, Tp, Fp = evaluation.calculate_metrics()
    
    # Display results
    evaluation.display_results()
    
    ####                          #### 
    ####          PRINTS          #### 
    ####                          ####
    
    # Print basic sequence characteristics
    print(f"Loaded scene: {base_folder}")
    print("Scene characteristics:")
    print(f"\t Aria RGB frames count: {len(img_timestamps_ns)}")
    print(f"\t Skeleton count: {len(gt_provider.get_skeleton_ids())}")
    
    # Print objects info
    # print(f"\t Static object count: {len(static_obj_ids)}")
    # print(f"\t Dynamic object count (tracked): {len(dynamic_obj_pose_cache)}")
    # print(f"\t Dynamic object count (tracked and moved - estimated): {len(dynamic_obj_moved)}")
    
    # Print results
    print(f"\t The ground truth objects are:", ground_truth)
    print(f"\t The LLM predictions are:", LLM_predictions)
    print(f"\t Correct Predictions (Tp): {Tp}")
    print(f"\t Total Predictions (Tp+Fp): {Tp+Fp}")
    print(f"\t Total LLM predictions: {len(LLM_predictions)}")
    print(f"\t Total Ground Truths: {len(ground_truth)}")
    
    # Print metrics
    print(f"\t Accuracy: {accuracy * 100:.2f}%")
    print(f"\t Precision: {precision * 100:.2f}%")
    print(f"\t Recall: {recall * 100:.2f}%")

if __name__ == "__main__":
    main()


