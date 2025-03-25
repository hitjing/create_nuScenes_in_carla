#!/usr/bin/env python

import carla
import pygame
import numpy as np
import cv2
import os
import json
import gzip
import math
from pathlib import Path
import threading
import argparse
import logging
import random
import h5py
import laspy
import random
from enum import Enum
import math

# ==============================================================================
# -- Constants -------------------------------------------------------------------
# ==============================================================================
DIS_CAR_SAVE = 50.0  # Maximum distance to save vehicles

# ==============================================================================
# -- Utility Functions --------------------------------------------------------
# ==============================================================================
save_interval = 20  # Save data every 'save_interval' frames

def calculate_velocity(actor):
    """
    Calculate the speed of an actor.
    """
    velocity = actor.get_velocity()
    return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

def compute_2d_distance(location1, location2):
    """
    Compute the 2D distance between two locations.
    """
    return math.sqrt((location1.x - location2.x)**2 + (location1.y - location2.y)**2)

def normalize_angle(angle):
    """
    Normalize angle to [-pi, pi].
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def calculate_cube_vertices(location, extent):
    """
    Calculate the 8 vertices of a cube given its location and extent.
    """
    x, y, z = extent.x, extent.y, extent.z
    return [
        (-x, -y, -z),
        (-x, -y, z),
        (-x, y, -z),
        (-x, y, z),
        (x, -y, -z),
        (x, -y, z),
        (x, y, -z),
        (x, y, z)
    ]

def build_projection_matrix(width, height, fov):
    """
    Build the camera's intrinsic matrix.
    """
    f = 0.5 * width / math.tan(0.5 * math.radians(fov))
    K = np.array([
        [f, 0, width / 2],
        [0, f, height / 2],
        [0, 0, 1]
    ])
    return K

def get_transform_matrix(transform):
    """
    Convert a carla.Transform to a 4x4 transformation matrix.
    """
    rotation = transform.rotation
    location = transform.location

    # Convert degrees to radians
    pitch = math.radians(rotation.pitch)
    yaw = math.radians(rotation.yaw)
    roll = math.radians(rotation.roll)

    # Rotation matrices
    R_x = np.array([
        [1, 0, 0, 0],
        [0, math.cos(pitch), -math.sin(pitch), 0],
        [0, math.sin(pitch), math.cos(pitch), 0],
        [0, 0, 0, 1]
    ])

    R_y = np.array([
        [math.cos(yaw), 0, math.sin(yaw), 0],
        [0, 1, 0, 0],
        [-math.sin(yaw), 0, math.cos(yaw), 0],
        [0, 0, 0, 1]
    ])

    R_z = np.array([
        [math.cos(roll), -math.sin(roll), 0, 0],
        [math.sin(roll), math.cos(roll), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    R = R_z @ R_y @ R_x

    # Translation matrix
    T = np.array([
        [1, 0, 0, location.x],
        [0, 1, 0, location.y],
        [0, 0, 1, location.z],
        [0, 0, 0, 1]
    ])

    return T @ R

def invert_transform_matrix(T):
    """
    Compute the inverse of a transformation matrix.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv

# ==============================================================================
# -- class RoadOption(Enum) ---------------------------------------------------
# ==============================================================================
class RoadOption(Enum):
    NONE = 0
    STRAIGHT = 1
    LEFT = 2
    RIGHT = 3

def normalize_angle_deg(angle):
    """
    Normalize angle to [-180, 180] degrees.
    """
    return (angle + 180) % 360 - 180

def determine_road_option(waypoint, distance=5.0):
    """
    Determine the road option based on the waypoint.
    """
    next_wps = waypoint.next(distance)
    if not next_wps:
        return RoadOption.NONE

    if len(next_wps) == 1:
        return RoadOption.STRAIGHT

    # Multiple next waypoints may indicate a turn
    current_yaw = waypoint.transform.rotation.yaw
    road_options = set()
    for next_wp in next_wps:
        target_yaw = next_wp.transform.rotation.yaw
        delta_yaw = normalize_angle_deg(target_yaw - current_yaw)
        if delta_yaw > 30:
            road_options.add(RoadOption.LEFT)
        elif delta_yaw < -30:
            road_options.add(RoadOption.RIGHT)
        else:
            road_options.add(RoadOption.STRAIGHT)

    if RoadOption.LEFT in road_options and RoadOption.RIGHT in road_options:
        return RoadOption.STRAIGHT  # Simplify to straight if multiple turns
    elif RoadOption.LEFT in road_options:
        return RoadOption.LEFT
    elif RoadOption.RIGHT in road_options:
        return RoadOption.RIGHT
    else:
        return RoadOption.STRAIGHT

def get_world_to_vehicle_transform(world_transform, vehicle_transform):
    """
    Compute the transformation matrix from world coordinates to a vehicle's coordinates.
    """
    T_world = get_transform_matrix(world_transform)
    T_vehicle = get_transform_matrix(vehicle_transform)
    T_world_to_vehicle = invert_transform_matrix(T_vehicle) @ T_world
    return T_world_to_vehicle

# ==============================================================================
# -- Sensor Interface and Environment Manager -------------------------------
# ==============================================================================
class SensorInterface:
    """
    Stores and manages sensor data.
    """
    def __init__(self):
        self.data = {}
        self.lock = threading.Lock()

    def update_data(self, sensor_id, data):
        with self.lock:
            self.data[sensor_id] = data

    def get_data(self):
        with self.lock:
            return self.data.copy()

class Env_Manager:
    """
    Manages the environment, including sensor setup, data collection, and saving.
    """
    def __init__(self, world, player, save_path='output'):
        self.world = world  # carla.World
        self.player = player  # ego vehicle: carla.Actor
        self.sensor_interface = SensorInterface()
        self.count = 0  # Frame index
        self.save_path = Path(save_path)
        self.setup_save_directories()
        self.setup_sensors()

        # Route planning
        self._global_plan = self.generate_route()
        self.navigation_idx = 0  # Current navigation index

    def generate_route(self):
        """
        Generate a predefined route as a global plan.
        """
        carla_map = self.world.get_map()
        spawn_points = carla_map.get_spawn_points()

        if not spawn_points:
            raise ValueError("No spawn points available in the map.")
        start_wp = carla_map.get_waypoint(spawn_points[0].location)

        global_plan = []
        current_wp = start_wp
        for _ in range(50):  # Generate 50 waypoints
            global_plan.append(current_wp)
            next_wps = current_wp.next(5.0)  # 5 meters ahead
            if next_wps:
                current_wp = next_wps[0]
            else:
                break  # Cannot proceed further
        return global_plan

    def get_target_commands(self, location, compass):
        """
        Get near and far commands based on the vehicle's current position.
        """
        if not self._global_plan:
            return [0.0, 0.0], 'None', [0.0, 0.0], 'None'

        current_wp = self.world.get_map().get_waypoint(location)
        closest_idx = min(range(len(self._global_plan)), key=lambda i: current_wp.transform.location.distance(self._global_plan[i].transform.location))

        near_idx = max(0, closest_idx)
        near_wp = self._global_plan[near_idx]
        try:
            near_command = determine_road_option(near_wp).name
        except AttributeError:
            near_command = 'None'
            print("road_option attribute not found for near_wp")

        far_idx = min(len(self._global_plan) - 1, near_idx + 5)  # e.g., 5 waypoints ahead
        far_wp = self._global_plan[far_idx]
        try:
            far_command = determine_road_option(far_wp).name
        except AttributeError:
            far_command = 'None'
            print("road_option attribute not found for far_wp")

        near_location = np.array([near_wp.transform.location.x, near_wp.transform.location.y], dtype=np.float32)
        far_location = np.array([far_wp.transform.location.x, far_wp.transform.location.y], dtype=np.float32)

        return far_location, far_command, near_location, near_command

    def evaluate_brake_conditions(self):
        """
        Set should_brake and only_ap_brake based on environment and vehicle state.
        """
        should_brake = False
        only_ap_brake = False

        # Get all traffic lights
        traffic_lights = self.world.get_actors().filter('*traffic_light*')
        ego_location = self.player.get_location()

        for tl in traffic_lights:
            tl_location = tl.get_transform().location
            dist = compute_2d_distance(ego_location, tl_location)
            if dist < 20.0:  # Within 20 meters
                if tl.state in [carla.TrafficLightState.Red, carla.TrafficLightState.Yellow]:
                    should_brake = True
                    only_ap_brake = True
                    break  # Brake for the first relevant traffic light

        # Additional conditions can be added here

        return should_brake, only_ap_brake

    def setup_save_directories(self):
        """
        Create directory structure for saving data.
        """
        scenario_name = "scenario_1"  # Can be parameterized
        town_id = self.world.get_map().name.split("/")[-1].replace("Town", "")
        weather_id = self.get_weather_id()
        route_id = "route_1"  # Can be parameterized or dynamically generated

        self.scenario_path = self.save_path / scenario_name / f"Town{town_id}_weather{weather_id}_route{route_id}"
        self.anno_path = self.scenario_path / 'anno'
        self.camera_path = self.scenario_path / 'camera'
        self.expert_assessment_path = self.scenario_path / 'expert_assessment'
        self.lidar_path = self.scenario_path / 'lidar'
        self.radar_path = self.scenario_path / 'radar'

        # Create directories
        self.anno_path.mkdir(parents=True, exist_ok=True)
        self.camera_path.mkdir(parents=True, exist_ok=True)
        self.expert_assessment_path.mkdir(parents=True, exist_ok=True)
        self.lidar_path.mkdir(parents=True, exist_ok=True)
        self.radar_path.mkdir(parents=True, exist_ok=True)

        # Define subdirectories for each camera type
        self.camera_types = [
            'depth_back', 'depth_back_left', 'depth_back_right',
            'depth_front', 'depth_front_left', 'depth_front_right',
            'instance_back', 'instance_back_left', 'instance_back_right',
            'instance_front', 'instance_front_left', 'instance_front_right',
            'rgb_back', 'rgb_back_left', 'rgb_back_right',
            'rgb_front', 'rgb_front_left', 'rgb_front_right',
            'rgb_top_down',
            'semantic_back', 'semantic_back_left', 'semantic_back_right',
            'semantic_front', 'semantic_front_left', 'semantic_front_right'
        ]

        for cam in self.camera_types:
            (self.camera_path / cam).mkdir(parents=True, exist_ok=True)

    def get_weather_id(self):
        """
        Get a weather identifier based on current weather parameters.
        """
        weather = self.world.get_weather()
        weather_id = f"cld{weather.cloudiness}_prec{weather.precipitation}_wind{weather.wind_intensity}"
        return weather_id

    def setup_sensors(self):
        """
        Set up and attach sensors to the ego vehicle.
        """
        blueprint_library = self.world.get_blueprint_library()

        # Define sensor configurations
        self.sensors = {}

        # Define different cameras with their transforms and name suffixes
        # 全新的 camera 配置列表（rgb、depth、semantic_seg、instance_seg）
        camera_configs = [
            # camera rgb
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.80, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_LEFT'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_RIGHT'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -2.0, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                    'width': 1600, 'height': 900, 'fov': 110,
                    'id': 'CAM_BACK'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_LEFT'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_RIGHT'
                    },
                # camera depth 
                {
                    'type': 'sensor.camera.depth',
                    'x': 0.80, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_DEPTH'
                    },
                {
                    'type': 'sensor.camera.depth',
                    'x': 0.27, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_LEFT_DEPTH'
                    },
                {
                    'type': 'sensor.camera.depth',
                    'x': 0.27, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_RIGHT_DEPTH'
                    },
                {
                    'type': 'sensor.camera.depth',
                    'x': -2.0, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                    'width': 1600, 'height': 900, 'fov': 110,
                    'id': 'CAM_BACK_DEPTH'
                    },
                {
                    'type': 'sensor.camera.depth',
                    'x': -0.32, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_LEFT_DEPTH'
                    },
                {
                    'type': 'sensor.camera.depth',
                    'x': -0.32, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_RIGHT_DEPTH'
                    },
                # camera seg
                {
                    'type': 'sensor.camera.semantic_segmentation',
                    'x': 0.80, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_SEM_SEG'
                    },
                {
                    'type': 'sensor.camera.semantic_segmentation',
                    'x': 0.27, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_LEFT_SEM_SEG'
                    },
                {
                    'type': 'sensor.camera.semantic_segmentation',
                    'x': 0.27, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_RIGHT_SEM_SEG'
                    },
                {
                    'type': 'sensor.camera.semantic_segmentation',
                    'x': -2.0, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                    'width': 1600, 'height': 900, 'fov': 110,
                    'id': 'CAM_BACK_SEM_SEG'
                    },
                {
                    'type': 'sensor.camera.semantic_segmentation',
                    'x': -0.32, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_LEFT_SEM_SEG'
                    },
                {
                    'type': 'sensor.camera.semantic_segmentation',
                    'x': -0.32, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_RIGHT_SEM_SEG'
                    },
                # camera seg
                {
                    'type': 'sensor.camera.instance_segmentation',
                    'x': 0.80, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_INS_SEG'
                    },
                {
                    'type': 'sensor.camera.instance_segmentation',
                    'x': 0.27, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_LEFT_INS_SEG'
                    },
                {
                    'type': 'sensor.camera.instance_segmentation',
                    'x': 0.27, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_RIGHT_INS_SEG'
                    },
                {
                    'type': 'sensor.camera.instance_segmentation',
                    'x': -2.0, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                    'width': 1600, 'height': 900, 'fov': 110,
                    'id': 'CAM_BACK_INS_SEG'
                    },
                {
                    'type': 'sensor.camera.instance_segmentation',
                    'x': -0.32, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_LEFT_INS_SEG'
                    },
                {
                    'type': 'sensor.camera.instance_segmentation',
                    'x': -0.32, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_RIGHT_INS_SEG'
                    },
                ### Debug sensor, not used by the model
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.0, 'y': 0.0, 'z': 50.0,
                    'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                    'width': 1600, 'height': 900, 'fov': 110,
                    'id': 'CAM_TOP_DOWN'
                    },
        ]

        # 创建camera传感器
        for cfg in camera_configs:
            bp = blueprint_library.find(cfg['type'])
            # 设置公用属性
            bp.set_attribute('image_size_x', str(cfg['width']))
            bp.set_attribute('image_size_y', str(cfg['height']))
            bp.set_attribute('fov', str(cfg['fov']))

            # 部分rgb相机若有image_type属性可设为Raw
            if 'rgb' in cfg['type'] and bp.has_attribute('image_type'):
                bp.set_attribute('image_type', 'Raw')

            # 构建transform
            transform = carla.Transform(
                carla.Location(x=cfg['x'], y=cfg['y'], z=cfg['z']),
                carla.Rotation(pitch=cfg['pitch'], yaw=cfg['yaw'], roll=cfg['roll'])
            )

            sensor_actor = self.world.spawn_actor(bp, transform, attach_to=self.player)
            sensor_actor.listen(lambda data, name=cfg['id']: self._on_camera_data(name, data))
            self.sensors[cfg['id']] = sensor_actor

        # Add LIDAR sensor
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '85')
        lidar_bp.set_attribute('rotation_frequency', '10')
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('points_per_second', '600000')
        lidar_bp.set_attribute('dropoff_general_rate', '0.0')
        lidar_bp.set_attribute('dropoff_intensity_limit', '0.0')
        lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
        lidar_transform = carla.Transform(carla.Location(x=-0.39, y=0.0, z=1.84))
        self.lidar_top = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.player)
        self.lidar_top.listen(lambda data: self._on_lidar_data('LIDAR_TOP', data))
        self.sensors['LIDAR_TOP'] = self.lidar_top

        # 5个雷达传感器
        radar_configs = [
            {
                'type': 'sensor.other.radar', 
                'x': 2.27, 'y': 0.0, 'z': 0.48, 
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'range': 100, 'horizontal_fov': 30, 'vertical_fov': 30,
                'id': 'RADAR_FRONT'
            },
            {
                'type': 'sensor.other.radar', 
                'x': 1.21, 'y': -0.85, 'z': 0.74, 
                'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
                'range': 100, 'horizontal_fov': 30, 'vertical_fov': 30,
                'id': 'RADAR_FRONT_LEFT'
            },
            {
                'type': 'sensor.other.radar', 
                'x': 1.21, 'y': 0.85, 'z': 0.74, 
                'roll': 0.0, 'pitch': 0.0, 'yaw': 90.0,
                'range': 100, 'horizontal_fov': 30, 'vertical_fov': 30,
                'id': 'RADAR_FRONT_RIGHT'
            },
            {
                'type': 'sensor.other.radar', 
                'x': -2.0, 'y': -0.67, 'z': 0.51, 
                'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
                'range': 100, 'horizontal_fov': 30, 'vertical_fov': 30,
                'id': 'RADAR_BACK_LEFT'
            },
            {
                'type': 'sensor.other.radar', 
                'x': -2.0, 'y': 0.67, 'z': 0.51, 
                'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
                'range': 100, 'horizontal_fov': 30, 'vertical_fov': 30,
                'id': 'RADAR_BACK_RIGHT'
            }
        ]

        for rcfg in radar_configs:
            radar_bp = blueprint_library.find(rcfg['type'])
            radar_bp.set_attribute('horizontal_fov', str(rcfg['horizontal_fov']))
            radar_bp.set_attribute('vertical_fov', str(rcfg['vertical_fov']))
            # points_per_second可根据需求设置更高，如100000
            radar_bp.set_attribute('points_per_second', '100000')
            radar_bp.set_attribute('range', str(rcfg['range']))
            radar_transform = carla.Transform(
                carla.Location(x=rcfg['x'], y=rcfg['y'], z=rcfg['z']),
                carla.Rotation(pitch=rcfg['pitch'], yaw=rcfg['yaw'], roll=rcfg['roll'])
            )
            radar_actor = self.world.spawn_actor(radar_bp, radar_transform, attach_to=self.player)
            radar_actor.listen(lambda data, name=rcfg['id']: self._on_radar_data(name, data))
            self.sensors[rcfg['id']] = radar_actor

        # Add GPS sensor
        gps_bp = blueprint_library.find('sensor.other.gnss')
        gps_transform = carla.Transform(carla.Location(x=-1.4, y=0.0, z=0.0))
        self.gps = self.world.spawn_actor(gps_bp, gps_transform, attach_to=self.player)
        self.gps.listen(lambda data: self._on_gps_data('GPS', data))
        self.sensors['GPS'] = self.gps

        # Add IMU sensor
        imu_bp = blueprint_library.find('sensor.other.imu')
        imu_transform = carla.Transform(carla.Location(x=-1.4, y=0.0, z=0.0), carla.Rotation(pitch=0.0, roll=0.0, yaw=0.0))
        self.imu = self.world.spawn_actor(imu_bp, imu_transform, attach_to=self.player)
        self.imu.listen(lambda data: self._on_imu_data('IMU', data))
        self.sensors['IMU'] = self.imu

    def get_trigger_volume(self, traffic_sign_actor):
        """
        获取交通标志的 trigger_volume 信息。
        """
        try:
            trigger_volume = traffic_sign_actor.trigger_volume
            return trigger_volume
        except AttributeError:
            # 如果交通标志没有 trigger_volume 属性，返回 None
            return None

    def _on_camera_data(self, sensor_id, image):
        """
        Handle camera data and save it.
        """
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        if 'depth' in sensor_id:
            # Process depth image
            array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
            depth = array[:, :, 0] * 256 + array[:, :, 1]
            depth = depth.astype(np.float32) / 65535.0 * 100.0  # Assume max depth of 100 meters
            self.sensor_interface.update_data(sensor_id, depth)
        elif 'semantic' in sensor_id:
            # Process semantic segmentation image
            array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
            self.sensor_interface.update_data(sensor_id, array)
        elif 'instance' in sensor_id:
            # Process instance segmentation image
            array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
            self.sensor_interface.update_data(sensor_id, array)
        else:
            # Process RGB image
            array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
            self.sensor_interface.update_data(sensor_id, array)

    def _on_lidar_data(self, sensor_id, data):
        """
        Handle LIDAR data and save it.
        """
        pts = np.frombuffer(data.raw_data, dtype=np.float32)
        pts = np.reshape(pts, (int(pts.shape[0]/4), 4))[:, :3]
        self.sensor_interface.update_data(sensor_id, pts)

    def _on_radar_data(self, sensor_id, data):
        """
        Handle RADAR data and save it.
        """
        points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
        self.sensor_interface.update_data(sensor_id, points)

    def _on_gps_data(self, sensor_id, data):
        """
        Handle GPS data and save it.
        """
        self.sensor_interface.update_data(sensor_id, [data.latitude, data.longitude, data.altitude])

    def _on_imu_data(self, sensor_id, data):
        """
        Handle IMU data and save it.
        """
        imu_data = np.array([
            data.accelerometer.x,
            data.accelerometer.y,
            data.accelerometer.z,
            data.gyroscope.x,
            data.gyroscope.y,
            data.gyroscope.z,
            data.compass
        ], dtype=np.float32)
        self.sensor_interface.update_data(sensor_id, imu_data)


    def tick(self):
        """
        Process sensor data and return results.
        """
        data = self.sensor_interface.get_data()
        if not data:
            return None  # Data not ready yet

        #self.count += 1  # Increment frame count

        # Extract IMU data
        imu_data = data.get('IMU', None)
        if imu_data is not None:
            acceleration = imu_data[:3].tolist()        # [acc_x, acc_y, acc_z]
            angular_velocity = imu_data[3:6].tolist()   # [gyro_x, gyro_y, gyro_z]
            compass = imu_data[6]                       # compass
        else:
            acceleration = [0.0, 0.0, 0.0]
            angular_velocity = [0.0, 0.0, 0.0]
            compass = 0.0

        # Extract GPS data
        gps_data = data.get('GPS', None)
        if gps_data is not None:
            gps_location = gps_data
        else:
            gps_location = [0.0, 0.0, 0.0]

        # Get vehicle's transform
        vehicle_location = self.player.get_location()
        vehicle_rotation = self.player.get_transform().rotation
        vehicle_theta = vehicle_rotation.yaw
        vehicle_transform = self.player.get_transform()

        # Get weather information
        weather = self.world.get_weather()
        weather_dict = {
            'cloudiness': getattr(weather, 'cloudiness', None),
            'precipitation': getattr(weather, 'precipitation', None),
            'precipitation_deposits': getattr(weather, 'precipitation_deposits', None),
            'wind_intensity': getattr(weather, 'wind_intensity', None),
            'sun_altitude_angle': getattr(weather, 'sun_altitude_angle', None),
            'sun_azimuth_angle': getattr(weather, 'sun_azimuth_angle', None),
            'fog_density': getattr(weather, 'fog_density', None),
            'fog_distance': getattr(weather, 'fog_distance', None),
            'wetness':getattr(weather,'wetness',None),
            'fog_falloff':getattr(weather,'fog_falloff',None)
        }

        # Get command information
        far_location, far_command, near_location, near_command = self.get_target_commands(vehicle_location, compass)

        # 定义命令映射表
        command_mapping = {
            'NONE': 0,
            'STRAIGHT': 1,
            'LEFT': 2,
            'RIGHT': 3
        }

        # Update navigation index based on distance to target waypoint
        if self.navigation_idx < len(self._global_plan) - 1:
            target_wp = self._global_plan[self.navigation_idx]
            dist_to_target = vehicle_location.distance(target_wp.transform.location)
            if dist_to_target < 5.0:  # If within 5 meters, move to next waypoint
                self.navigation_idx += 1

        # 如果navigation_idx更新后需要重新获取指令
        if self.navigation_idx < len(self._global_plan):
            far_wp = self._global_plan[self.navigation_idx]
            far_location = [far_wp.transform.location.x, far_wp.transform.location.y]
            far_road_option = determine_road_option(far_wp)
            far_command = command_mapping[far_road_option.name]

            # Get the previous waypoint as near command
            if self.navigation_idx > 0:
                near_wp = self._global_plan[self.navigation_idx - 1]
            else:
                near_wp = far_wp
            near_road_option = determine_road_option(near_wp)
            near_command = command_mapping[near_road_option.name]
        else:
            far_location = [0.0, 0.0]
            far_command = 0
            near_location = [0.0, 0.0]
            near_command = 0

        # 定义next_command，这里简单地使用far_command作为next_command的值
        next_command = far_command

        # Set braking conditions
        should_brake, only_ap_brake = self.evaluate_brake_conditions()

        # Get bounding boxes
        bounding_boxes = self.get_bounding_boxes(lidar=data.get('LIDAR_TOP', None))

         # 将所有radar数据放入一个字典中
        radar_data_dict = {}
        for radar_id in ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']:
            radar_data_dict[radar_id] = data.get(radar_id, None)

        # Compute world to ego transformation matrix
        T_ego = get_transform_matrix(vehicle_transform)
        T_world_to_ego = invert_transform_matrix(T_ego)

        # Compute world to vehicle transformations
        world_to_vehicle_dict = {}
        for actor in self.world.get_actors():
            if 'vehicle' in actor.type_id and actor.id != self.player.id:
                actor_transform = actor.get_transform()
                T_vehicle = get_transform_matrix(actor_transform)
                T_world_to_vehicle = invert_transform_matrix(T_vehicle)
                world_to_vehicle_dict[actor.id] = T_world_to_vehicle.tolist()

        # Construct the annotation dictionary with the specified structure
        annotation = {
            'x': vehicle_location.x,
            'y': vehicle_location.y,
            'throttle': self.player.get_control().throttle,
            'steer': self.player.get_control().steer,
            'brake': self.player.get_control().brake,
            'reverse': self.player.get_control().reverse,
            'theta': vehicle_theta,
            'speed': self.get_forward_speed(),
            'x_command_far': float(far_location[0]),
            'y_command_far': float(far_location[1]),
            'command_far': far_command,
            'x_command_near': float(near_location[0]),
            'y_command_near': float(near_location[1]),
            'command_near': near_command,
            'should_brake': should_brake,
            'only_ap_brake': only_ap_brake,
            'x_target': float(far_location[0]),
            'y_target': float(far_location[1]),
            'next_command': next_command,  # Needs further definition
            'weather': weather_dict,
            'acceleration': acceleration,       # Acceleration from IMU
            'angular_velocity': angular_velocity,  # Angular velocity from IMU
            'sensors': self.get_sensors_anno(),
            'bounding_boxes': bounding_boxes,
            'world2ego': T_world_to_ego.tolist()
        }

        # Save sensor transforms
        self.fill_sensor_transforms(annotation, T_world_to_ego)

        return {
            'annotation': annotation,
            'camera_data': {
                'depth_back': data.get('CAM_BACK_DEPTH', None),
                'depth_back_left': data.get('CAM_BACK_LEFT_DEPTH', None),
                'depth_back_right': data.get('CAM_BACK_RIGHT_DEPTH', None),
                'depth_front': data.get('CAM_FRONT_DEPTH', None),
                'depth_front_left': data.get('CAM_FRONT_LEFT_DEPTH', None),
                'depth_front_right': data.get('CAM_FRONT_RIGHT_DEPTH', None),
                'instance_back': data.get('CAM_BACK_INS_SEG', None),
                'instance_back_left': data.get('CAM_BACK_LEFT_INS_SEG', None),
                'instance_back_right': data.get('CAM_BACK_RIGHT_INS_SEG', None),
                'instance_front': data.get('CAM_FRONT_INS_SEG', None),
                'instance_front_left': data.get('CAM_FRONT_LEFT_INS_SEG', None),
                'instance_front_right': data.get('CAM_FRONT_RIGHT_INS_SEG', None),
                'rgb_back': data.get('CAM_BACK', None),
                'rgb_back_left': data.get('CAM_BACK_LEFT', None),
                'rgb_back_right': data.get('CAM_BACK_RIGHT', None),
                'rgb_front': data.get('CAM_FRONT', None),
                'rgb_front_left': data.get('CAM_FRONT_LEFT', None),
                'rgb_front_right': data.get('CAM_FRONT_RIGHT', None),
                'rgb_top_down': data.get('CAM_TOP_DOWN', None),
                'semantic_back': data.get('CAM_BACK_SEM_SEG', None),
                'semantic_back_left': data.get('CAM_BACK_LEFT_SEM_SEG', None),
                'semantic_back_right': data.get('CAM_BACK_RIGHT_SEM_SEG', None),
                'semantic_front': data.get('CAM_FRONT_SEM_SEG', None),
                'semantic_front_left': data.get('CAM_FRONT_LEFT_SEM_SEG', None),
                'semantic_front_right': data.get('CAM_FRONT_RIGHT_SEM_SEG', None)
            },
            'lidar_data': data.get('LIDAR_TOP', None),
            'radar_data': radar_data_dict,
            'gps_data': gps_location
        }

    def fill_sensor_transforms(self, annotation, world2ego):
        """
        Fill in sensor transformation matrix information.
        """
        sensor_transforms = {}
        for sensor_name, sensor in self.sensors.items():
            config = self.get_sensor_config(sensor_name)
        
            # Convert lists to numpy arrays for matrix operations
            sensor2ego = np.array(config.get('sensor2ego', np.eye(4)))
        
            # Compute world2sensor transformation matrix
            world2sensor = np.matmul(world2ego, sensor2ego)
        
            if sensor_name.startswith('CAM_'):
                sensor_transforms[sensor_name] = {
                    'location': config['sensor_location'],
                    'rotation': config['sensor_rotation'],
                    'intrinsic': config['intrinsic'],
                    'world2cam': world2sensor.tolist(),
                    'cam2ego': config['sensor2ego'],
                    'fov': config.get('fov', None),
                    'image_size_x': config.get('image_size_x', 1600),
                    'image_size_y': config.get('image_size_y', 900)
                }
            elif sensor_name.startswith('RADAR_'):
                sensor_transforms[sensor_name] = {
                    'location': config['sensor_location'],
                    'rotation': config['sensor_rotation'],
                    'world2radar': world2sensor.tolist(),
                    'radar2ego': config['sensor2ego']
                }
            elif sensor_name.startswith('LIDAR_'):
                sensor_transforms[sensor_name] = {
                    'location': config['sensor_location'],
                    'rotation': config['sensor_rotation'],
                    'world2lidar': world2sensor.tolist(),
                    'lidar2ego': config['sensor2ego']
                }
            else:
                # Handle other sensor types if necessary
                sensor_transforms[sensor_name] = {
                    'location': config['sensor_location'],
                    'rotation': config['sensor_rotation'],
                    'transformation': world2sensor.tolist()
                }
    
        annotation['sensors'] = sensor_transforms

    def get_forward_speed(self):
        """
        Get the forward speed of the vehicle in km/h.
        """
        velocity = self.player.get_velocity()
        return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # m/s to km/h
    
    def get_light_state(self, actor):
        """
        获取车辆或交通灯的灯光状态。
        若为车辆，则调用 actor.get_light_state() 返回车辆灯光状态（如Brake、LeftBlinker等）。
        若为交通灯，则 actor.state 为 carla.TrafficLightState 返回红黄绿。
        """
        # 判断actor类型
        if 'vehicle' in actor.type_id:
            # 对车辆而言：try获取light state
            try:
                light_state = actor.get_light_state()  # 返回carla.VehicleLightState
                # 将light_state转化为字符串
                # light_state是一个枚举值的组合，需要判断其bitmask
                ls_str = []
                if light_state & carla.VehicleLightState.Brake:
                    ls_str.append('Brake')
                if light_state & carla.VehicleLightState.LeftBlinker:
                    ls_str.append('LeftBlinker')
                if light_state & carla.VehicleLightState.RightBlinker:
                    ls_str.append('RightBlinker')
                if light_state & carla.VehicleLightState.LowBeam:
                    ls_str.append('LowBeam')
                if light_state & carla.VehicleLightState.HighBeam:
                    ls_str.append('HighBeam')
                if light_state & carla.VehicleLightState.Reverse:
                    ls_str.append('Reverse')
                if light_state & carla.VehicleLightState.Fog:
                    ls_str.append('Fog')
                if light_state & carla.VehicleLightState.Position:
                    ls_str.append('Position')
                if light_state & carla.VehicleLightState.Interior:
                    ls_str.append('Interior')
                if len(ls_str) == 0:
                    return 'None'
                return ','.join(ls_str)
            except:
                return 'None'
        elif 'traffic_light' in actor.type_id:
            # 对交通灯而言
            tl_state = actor.state  # carla.TrafficLightState
            if tl_state == carla.TrafficLightState.Green:
                return 'green'
            elif tl_state == carla.TrafficLightState.Yellow:
                return 'yellow'
            elif tl_state == carla.TrafficLightState.Red:
                return 'red'
            else:
                return 'unknown'
        else:
            # 对其他类型actor，没有light_state
            return 'None'
        
    def get_road_id(self, location):
        """
        根据 location 获取对应道路ID。
        通过 waypoint = CarlaDataProvider.get_map().get_waypoint(location) 获取waypoint，再返回waypoint.road_id。
        """
        waypoint = self.world.get_map().get_waypoint(location)
        return waypoint.road_id

    def get_lane_id(self, location):
        """
        根据 location 获取对应车道ID。
        """
        waypoint = self.world.get_map().get_waypoint(location)
        return waypoint.lane_id

    def get_section_id(self, location):
        """
        根据 location 获取对应路段(section) ID。
        部分CARLA版本中waypoint并没有section_id属性，如无请根据实际情况返回默认值或额外查表。
        假设本案例中waypoint有section_id属性，否则可返回-1或None。
        """
        waypoint = self.world.get_map().get_waypoint(location)
        # 如果waypoint没有section_id属性，可尝试:
        # return getattr(waypoint, 'section_id', -1)
        # 本示例假设有该属性:
        return getattr(waypoint, 'section_id', -1)
    
    def get_lidar_points_in_bbox(self, center, extent, lidar_points):
        """
        计算激光雷达点云中落在边界框内的点数
        """
        if lidar_points is None:
            return -1
    
        # 定义边界框范围
        x_min = center.x - extent.x
        x_max = center.x + extent.x
        y_min = center.y - extent.y
        y_max = center.y + extent.y
        z_min = center.z - extent.z
        z_max = center.z + extent.z
    
        # 过滤点云
        points_in_bbox = lidar_points[
            (lidar_points[:,0] >= x_min) & (lidar_points[:,0] <= x_max) &
            (lidar_points[:,1] >= y_min) & (lidar_points[:,1] <= y_max) &
            (lidar_points[:,2] >= z_min) & (lidar_points[:,2] <= z_max)
        ]
    
        return points_in_bbox.shape[0]

    def get_bounding_boxes(self, lidar=None):
        """
        获取场景中所有相关演员的边界框，作为一个列表返回。
        确保 ego_vehicle 始终位于列表首位。
        """
        bounding_boxes = []
        actors = self.world.get_actors()
        ego = self.player

        # 定义交通信号灯状态映射
        traffic_light_state_mapping = {
            carla.TrafficLightState.Red: 0,
            carla.TrafficLightState.Yellow: 1,
            carla.TrafficLightState.Green: 2,
            carla.TrafficLightState.Off: 3,
            carla.TrafficLightState.Unknown: -1
        }

        # 首先处理 ego_vehicle
        if ego and ego.is_alive:
            actor = ego
            actor_transform = actor.get_transform()
            actor_location = actor_transform.location
            actor_rotation = actor_transform.rotation
            extent = actor.bounding_box.extent
                    # 检查边界框尺寸
            if extent.x <= 0 or extent.y <= 0 or extent.z <= 0:
                logging.warning(f"Invalid bounding box size for ego_actor {actor.id}: "
                                f"length={extent.x*2}, width={extent.y*2}, height={extent.z*2}")
            center = actor_transform.transform(actor.bounding_box.location)
            local_verts = calculate_cube_vertices(actor.bounding_box.location, actor.bounding_box.extent)
            global_verts = []
            for lv in local_verts:
                g_v = actor_transform.transform(carla.Location(x=lv[0], y=lv[1], z=lv[2]))
                global_verts.append([g_v.x, g_v.y, g_v.z])

            # 计算速度（单位：km/h）
            speed = self.get_forward_speed()

            # 构建 ego_vehicle 的边界框字典
            bbox = {
                'class': 'ego_vehicle',
                'id': str(actor.id),
                'location': [actor_location.x, actor_location.y, actor_location.z],
                'rotation': [actor_rotation.pitch, actor_rotation.yaw, actor_rotation.roll],
                'center': [center.x, center.y, center.z],
                'extent': [extent.x, extent.y, extent.z],
                'semantic_tags': actor.semantic_tags if hasattr(actor, 'semantic_tags') else [],
                'type_id': actor.type_id,
                'road_id': self.get_road_id(actor.get_location()),
                'lane_id': self.get_lane_id(actor.get_location()),
                'section_id': self.get_section_id(actor.get_location()),
                'color': self.get_actor_color(actor),
                'base_type': 'car',
                'world_cord': global_verts,
                'bbx_loc': [actor.bounding_box.location.x, actor.bounding_box.location.y, actor.bounding_box.location.z],
                'speed': speed,
                'brake': self.player.get_control().brake,
                'world2ego': get_world_to_vehicle_transform(actor_transform, self.player.get_transform()).tolist()
            }

            # 确保 semantic_tags 是一个扁平的整数列表
            if isinstance(bbox['semantic_tags'], list) and all(isinstance(i, list) for i in bbox['semantic_tags']):
                bbox['semantic_tags'] = [tag for sublist in bbox['semantic_tags'] for tag in sublist]
            elif isinstance(bbox['semantic_tags'], list) and not all(isinstance(i, list) for i in bbox['semantic_tags']):
                # 已经是扁平列表
                pass
            else:
                # 如果 semantic_tags 不是列表，将其转换为列表
                bbox['semantic_tags'] = [bbox['semantic_tags']]

            bounding_boxes.append(bbox)  # 添加到列表首位

        # 然后处理其他演员
        for actor in actors:
            if not actor.is_alive or actor.id == ego.id:
                continue

            actor_transform = actor.get_transform()
            actor_location = actor_transform.location
            actor_rotation = actor_transform.rotation
            extent = actor.bounding_box.extent
            
            # 检查边界框尺寸
            if extent.x <= 0 or extent.y <= 0 or extent.z <= 0:
                logging.warning(f"Invalid bounding box size for actor {actor.id}: "
                                f"length={extent.x*2}, width={extent.y*2}, height={extent.z*2}")
                continue  # 跳过无效的边界框

            center = actor_transform.transform(actor.bounding_box.location)
            local_verts = calculate_cube_vertices(actor.bounding_box.location, actor.bounding_box.extent)
            global_verts = []
            for lv in local_verts:
                g_v = actor_transform.transform(carla.Location(x=lv[0], y=lv[1], z=lv[2]))
                global_verts.append([g_v.x, g_v.y, g_v.z])

            dist = compute_2d_distance(actor.get_location(), ego.get_location())
            if dist > DIS_CAR_SAVE:
                continue  # 跳过距离过远的演员

            # 确定演员的类别
            if 'vehicle' in actor.type_id:
                actor_class = 'vehicle'
            elif 'walker' in actor.type_id:
                actor_class = 'walker'
            elif 'traffic_light' in actor.type_id:
                actor_class = 'traffic_light'
            elif ('traffic.speed_limit' in actor.type_id) or ('traffic.stop' in actor.type_id) or ('traffic.yield' in actor.type_id):
                actor_class = 'traffic_sign'
            else:
                continue  # 跳过其他类型的演员

            # 通用属性
            bbox = {
                'class': actor_class,
                'id': str(actor.id),
                'location': [actor_location.x, actor_location.y, actor_location.z],
                'rotation': [actor_rotation.pitch, actor_rotation.yaw, actor_rotation.roll],
                'center': [center.x, center.y, center.z],
                'extent': [extent.x, extent.y, extent.z],
                'semantic_tags': actor.semantic_tags if hasattr(actor, 'semantic_tags') else [],
                'type_id': actor.type_id,
                'road_id': self.get_road_id(actor.get_location()),
                'lane_id': self.get_lane_id(actor.get_location()),
                'section_id': self.get_section_id(actor.get_location())
            }

            # 确保 semantic_tags 是一个扁平的整数列表
            if isinstance(bbox['semantic_tags'], list) and all(isinstance(i, list) for i in bbox['semantic_tags']):
                bbox['semantic_tags'] = [tag for sublist in bbox['semantic_tags'] for tag in sublist]
            elif isinstance(bbox['semantic_tags'], list) and not all(isinstance(i, list) for i in bbox['semantic_tags']):
                # 已经是扁平列表
                pass
            else:
                # 如果 semantic_tags 不是列表，将其转换为列表
                bbox['semantic_tags'] = [bbox['semantic_tags']]

            # 根据类别添加额外属性
            if actor_class == 'vehicle':
                state = 'dynamic' if calculate_velocity(actor) > 0.1 else 'static'
                bbox['state'] = state
                bbox['distance'] = dist
                bbox['color'] = self.get_actor_color(actor)
                if actor.type_id == 'vehicle.ford.ambulance' or actor.type_id == 'vehicle.carlamotors.carlacola':
                    bbox['base_type'] = 'van'
                elif actor.type_id == 'vehicle.carlamotors.firetruck':
                    bbox['base_type'] = 'truck'
                elif actor.type_id == 'vehicle.bh.crossbike' or actor.type_id == 'vehicle.diamondback.century' or actor.type_id == 'vehicle.gazelle.omafiets':
                    bbox['base_type'] = 'bicycle'
                else:
                    bbox['base_type'] = 'car'
                bbox['brake'] = self.player.get_control().brake
                bbox['world_cord'] = global_verts
                bbox['bbx_loc'] = [actor.bounding_box.location.x, actor.bounding_box.location.y, actor.bounding_box.location.z]
                bbox['speed'] = calculate_velocity(actor)
                bbox['light_state'] = self.get_light_state(actor)
                # LIDAR 点云在边界框内的数量
                if lidar is not None:
                    bbox['num_points'] = self.get_lidar_points_in_bbox(center, extent, lidar)
                else:
                    bbox['num_points'] = -1
                bbox['world2vehicle'] = get_world_to_vehicle_transform(actor_transform, self.player.get_transform()).tolist()

            elif actor_class == 'walker':
                bbox['gender'] = getattr(actor, 'gender', 'unknown')  # 假设存在 'gender' 属性
                bbox['age'] = getattr(actor, 'age', 'unknown')        # 假设存在 'age' 属性
                bbox['bone'] = getattr(actor, 'bone', 'unknown')      # 假设存在 'bone' 属性
                bbox['world2ped'] = get_world_to_vehicle_transform(actor_transform, self.player.get_transform()).tolist()
                # LIDAR 点云在边界框内的数量
                if lidar is not None:
                    bbox['num_points'] = self.get_lidar_points_in_bbox(center, extent, lidar)
                else:
                    bbox['num_points'] = -1
                bbox['bbx_loc'] = [actor.bounding_box.location.x, actor.bounding_box.location.y, actor.bounding_box.location.z]
                bbox['world_cord'] = global_verts
                bbox['distance'] = dist
                bbox['speed'] = calculate_velocity(actor)

            elif actor_class == 'traffic_light':
                # 转换交通信号灯状态为整数
                tl_state = actor.state
                bbox['state'] = traffic_light_state_mapping.get(tl_state, -1)
                bbox['affects_ego'] = self.is_traffic_light_affecting_ego(actor)
                bbox['distance'] = dist
                # 尝试获取 trigger_volume 信息
                trigger_volume = self.get_trigger_volume(actor)
                if trigger_volume:
                    bbox['trigger_volume_location'] = [
                        trigger_volume.location.x,
                        trigger_volume.location.y,
                        trigger_volume.location.z
                    ]
                    bbox['trigger_volume_rotation'] = [
                        trigger_volume.rotation.pitch,
                        trigger_volume.rotation.yaw,
                        trigger_volume.rotation.roll
                    ]
                    bbox['trigger_volume_extent'] = [
                        trigger_volume.extent.x,
                        trigger_volume.extent.y,
                        trigger_volume.extent.z
                    ]
                else:
                    # 如果无法获取 trigger_volume，使用占位符
                    bbox['trigger_volume_location'] = [0.0, 0.0, 0.0]
                    bbox['trigger_volume_rotation'] = [0.0, 0.0, 0.0]
                    bbox['trigger_volume_extent'] = [0.0, 0.0, 0.0]

            elif actor_class == 'traffic_sign':
                # 获取交通标志的触发状态
                bbox['affects_ego'] = self.is_traffic_sign_affecting_ego(actor)
                bbox['world2sign'] = get_world_to_vehicle_transform(actor_transform, self.player.get_transform()).tolist()
                bbox['distance'] = dist
                bbox['bbx_loc'] = [actor.bounding_box.location.x, actor.bounding_box.location.y, actor.bounding_box.location.z]
                # 获取 trigger_volume 信息
                trigger_volume = self.get_trigger_volume(actor)
                if trigger_volume:
                    bbox['trigger_volume_location'] = [
                        trigger_volume.location.x,
                        trigger_volume.location.y,
                        trigger_volume.location.z
                    ]
                    bbox['trigger_volume_rotation'] = [
                        trigger_volume.rotation.pitch,
                        trigger_volume.rotation.yaw,
                        trigger_volume.rotation.roll
                    ]
                    bbox['trigger_volume_extent'] = [
                        trigger_volume.extent.x,
                        trigger_volume.extent.y,
                        trigger_volume.extent.z
                    ]
                else:
                    # 如果无法获取 trigger_volume，使用占位符
                    bbox['trigger_volume_location'] = [0.0, 0.0, 0.0]
                    bbox['trigger_volume_rotation'] = [0.0, 0.0, 0.0]
                    bbox['trigger_volume_extent'] = [0.0, 0.0, 0.0]

            bounding_boxes.append(bbox)

        return bounding_boxes



    def is_traffic_light_affecting_ego(self, traffic_light):
        """
        Determine if the traffic light affects the ego vehicle.
        Placeholder implementation; needs to be defined based on specific criteria.
        """
        # Implement logic to determine if traffic light affects ego
        return True  # Placeholder

    def is_traffic_sign_affecting_ego(self, traffic_sign):
        """
        Determine if the traffic sign affects the ego vehicle.
        Placeholder implementation; needs to be defined based on specific criteria.
        """
        # Implement logic to determine if traffic sign affects ego
        return True  # Placeholder

    def get_actor_color(self, actor):
        """
        Get the color of an actor. Returns white if not available.
        """
        default_color = "0,0,0"  # White
        try:
            color = getattr(actor, 'color', None)
            if color is not None:
                return color
            else:
                return default_color
        except AttributeError:
            return default_color

    def get_sensors_anno(self):
        """
        Get annotations for all sensors.
        """
        results = {}
        for name_suffix, sensor in self.sensors.items():
            transform = sensor.get_transform()
            location = [transform.location.x, transform.location.y, transform.location.z]
            rotation = [transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll]

            config = self.get_sensor_config(name_suffix)
            intrinsic = config.get('intrinsic', None)

            if name_suffix.startswith('CAM_'):
                results[name_suffix] = {
                    'location': location,
                    'rotation': rotation,
                    'intrinsic': intrinsic,
                    'world2cam': config.get('world2sensor', []),
                    'cam2ego': config.get('sensor2ego', []),
                    'fov': config.get('fov', None),
                    'image_size_x': config.get('image_size_x', 1600),
                    'image_size_y': config.get('image_size_y', 900)
                }
            elif name_suffix.startswith('RADAR_'):
                results[name_suffix] = {
                    'location': location,
                    'rotation': rotation,
                    'world2radar': config.get('world2sensor', []),
                    'radar2ego': config.get('sensor2ego', [])
                }
            elif name_suffix.startswith('LIDAR_'):
                results[name_suffix] = {
                    'location': location,
                    'rotation': rotation,
                    'world2lidar': config.get('world2sensor', []),
                    'lidar2ego': config.get('sensor2ego', [])
                }
            else:
                # Handle other sensor types if necessary
                results[name_suffix] = {}

        return results

    def get_sensor_config(self, sensor_name):
        """
        Get the transformation configuration for a sensor.
        """
        # Define relative transforms for each sensor
        sensor_relative_transforms = {
            # Camera RGB
            'CAM_FRONT': carla.Transform(carla.Location(x=0.80, y=0.0, z=1.60), carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)),
            'CAM_FRONT_LEFT': carla.Transform(carla.Location(x=0.27, y=-0.55, z=1.60), carla.Rotation(pitch=0.0, yaw=-55.0, roll=0.0)),
            'CAM_FRONT_RIGHT': carla.Transform(carla.Location(x=0.27, y=0.55, z=1.60), carla.Rotation(pitch=0.0, yaw=55.0, roll=0.0)),
            'CAM_BACK': carla.Transform(carla.Location(x=-2.0, y=0.0, z=1.60), carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0)),
            'CAM_BACK_LEFT': carla.Transform(carla.Location(x=-0.32, y=-0.55, z=1.60), carla.Rotation(pitch=0.0, yaw=-110.0, roll=0.0)),
            'CAM_BACK_RIGHT': carla.Transform(carla.Location(x=-0.32, y=0.55, z=1.60), carla.Rotation(pitch=0.0, yaw=110.0, roll=0.0)),
            'CAM_TOP_DOWN': carla.Transform(carla.Location(x=0.0, y=0.0, z=50.0), carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)),

            # Camera Depth
            'CAM_FRONT_DEPTH': carla.Transform(carla.Location(x=0.80, y=0.0, z=1.60), carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)),
            'CAM_FRONT_LEFT_DEPTH': carla.Transform(carla.Location(x=0.27, y=-0.55, z=1.60), carla.Rotation(pitch=0.0, yaw=-55.0, roll=0.0)),
            'CAM_FRONT_RIGHT_DEPTH': carla.Transform(carla.Location(x=0.27, y=0.55, z=1.60), carla.Rotation(pitch=0.0, yaw=55.0, roll=0.0)),
            'CAM_BACK_DEPTH': carla.Transform(carla.Location(x=-2.0, y=0.0, z=1.60), carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0)),
            'CAM_BACK_LEFT_DEPTH': carla.Transform(carla.Location(x=-0.32, y=-0.55, z=1.60), carla.Rotation(pitch=0.0, yaw=-110.0, roll=0.0)),
            'CAM_BACK_RIGHT_DEPTH': carla.Transform(carla.Location(x=-0.32, y=0.55, z=1.60), carla.Rotation(pitch=0.0, yaw=110.0, roll=0.0)),

            # Camera Semantic Segmentation
            'CAM_FRONT_SEM_SEG': carla.Transform(carla.Location(x=0.80, y=0.0, z=1.60), carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)),
            'CAM_FRONT_LEFT_SEM_SEG': carla.Transform(carla.Location(x=0.27, y=-0.55, z=1.60), carla.Rotation(pitch=0.0, yaw=-55.0, roll=0.0)),
            'CAM_FRONT_RIGHT_SEM_SEG': carla.Transform(carla.Location(x=0.27, y=0.55, z=1.60), carla.Rotation(pitch=0.0, yaw=55.0, roll=0.0)),
            'CAM_BACK_SEM_SEG': carla.Transform(carla.Location(x=-2.0, y=0.0, z=1.60), carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0)),
            'CAM_BACK_LEFT_SEM_SEG': carla.Transform(carla.Location(x=-0.32, y=-0.55, z=1.60), carla.Rotation(pitch=0.0, yaw=-110.0, roll=0.0)),
            'CAM_BACK_RIGHT_SEM_SEG': carla.Transform(carla.Location(x=-0.32, y=0.55, z=1.60), carla.Rotation(pitch=0.0, yaw=110.0, roll=0.0)),

            # Camera Instance Segmentation
            'CAM_FRONT_INS_SEG': carla.Transform(carla.Location(x=0.80, y=0.0, z=1.60), carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)),
            'CAM_FRONT_LEFT_INS_SEG': carla.Transform(carla.Location(x=0.27, y=-0.55, z=1.60), carla.Rotation(pitch=0.0, yaw=-55.0, roll=0.0)),
            'CAM_FRONT_RIGHT_INS_SEG': carla.Transform(carla.Location(x=0.27, y=0.55, z=1.60), carla.Rotation(pitch=0.0, yaw=55.0, roll=0.0)),
            'CAM_BACK_INS_SEG': carla.Transform(carla.Location(x=-2.0, y=0.0, z=1.60), carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0)),
            'CAM_BACK_LEFT_INS_SEG': carla.Transform(carla.Location(x=-0.32, y=-0.55, z=1.60), carla.Rotation(pitch=0.0, yaw=-110.0, roll=0.0)),
            'CAM_BACK_RIGHT_INS_SEG': carla.Transform(carla.Location(x=-0.32, y=0.55, z=1.60), carla.Rotation(pitch=0.0, yaw=110.0, roll=0.0)),
            
            # LIDAR and RADAR are handled separately
            'LIDAR_TOP': carla.Transform(carla.Location(x=-0.39, y=0.0, z=1.84), carla.Rotation(pitch=0, yaw=0, roll=0)),
            # RADAR
            'RADAR_FRONT': carla.Transform(carla.Location(x=2.27, y=0.0, z=0.48), carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)),
            'RADAR_FRONT_LEFT': carla.Transform(carla.Location(x=1.21, y=-0.85, z=0.74), carla.Rotation(pitch=0.0, yaw=-90.0, roll=0.0)),
            'RADAR_FRONT_RIGHT': carla.Transform(carla.Location(x=1.21, y=0.85, z=0.74), carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0)),
            'RADAR_BACK_LEFT': carla.Transform(carla.Location(x=-2.0, y=-0.67, z=0.51), carla.Rotation(pitch=0.0, yaw=-90.0, roll=0.0)),
            'RADAR_BACK_RIGHT': carla.Transform(carla.Location(x=-2.0, y=0.67, z=0.51), carla.Rotation(pitch=0.0, yaw=-90.0, roll=0.0)),

            'GPS': carla.Transform(carla.Location(x=-1.4, y=0.0, z=0.0), carla.Rotation(pitch=0, yaw=0, roll=0)),
            'IMU': carla.Transform(carla.Location(x=-1.4, y=0.0, z=0.0), carla.Rotation(pitch=0, yaw=0, roll=0))
        }

        if sensor_name not in sensor_relative_transforms:
            logging.warning(f"Sensor '{sensor_name}' not found in sensor_relative_transforms.")
            return {
                'sensor2ego': np.eye(4).tolist(),
                'ego2sensor': np.eye(4).tolist(),
                'sensor_location': [0.0, 0.0, 0.0],
                'sensor_rotation': [0.0, 0.0, 0.0],
                'intrinsic': None,
                'fov': None,
                'image_size_x': 1600,
                'image_size_y': 900
            }

        # Get relative transform
        transform = sensor_relative_transforms[sensor_name]

        # Compute sensor to ego transformation
        sensor2ego = get_transform_matrix(transform)
        ego2sensor = invert_transform_matrix(sensor2ego)

        # Compute intrinsic matrix for cameras
        if sensor_name.startswith('CAM_'):
            intrinsic = build_projection_matrix(1600, 900, 70).tolist()
            fov = 70
            image_size_x = 1600
            image_size_y = 900
        else:
            intrinsic = None
            fov = None
            image_size_x = 0
            image_size_y = 0

        return {
            'sensor2ego': sensor2ego.tolist(),
            'ego2sensor': ego2sensor.tolist(),
            'sensor_location': [transform.location.x, transform.location.y, transform.location.z],
            'sensor_rotation': [transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll],
            'intrinsic': intrinsic,
            'fov': fov,
            'image_size_x': image_size_x,
            'image_size_y': image_size_y
        }

    def save(self, results):
        """
        Save sensor data and annotations.
        """
        annotation = results['annotation']
        camera_data = results['camera_data']
        lidar_data = results['lidar_data']
        radar_data = results['radar_data']
        gps_data = results['gps_data']
        frame = self.count
        frame_str = f'{frame:05}'

        # Save annotation as JSON.gz
        anno_file = self.anno_path / f'{frame_str}.json.gz'
        with gzip.open(anno_file, 'wt', encoding='utf-8') as f:
            json.dump(annotation, f, indent=4)

        # Save camera data
        for cam_type, img in camera_data.items():
            if img is not None:
                if 'rgb' in cam_type:
                    # Save as JPG
                    img_filename = self.camera_path / cam_type / f'{frame_str}.jpg'
                    cv2.imwrite(str(img_filename), img)
                elif 'depth' in cam_type or 'instance' in cam_type or 'semantic' in cam_type:
                    # Save depth images as PNG
                    img_filename = self.camera_path / cam_type / f'{frame_str}.png'
                    cv2.imwrite(str(img_filename), img)
                
        # 将五个雷达数据统一保存到同一个H5文件中
        # 在调用save函数时，results就是tick函数的返回值
        radar_data_dict = results['radar_data']  # 从results中获取radar_data字典
        radar_filename = self.radar_path / f'{frame_str}.h5'
        with h5py.File(str(radar_filename), 'w') as hf:
            # 假设数据为 np.ndarray 类型
            # 为每个雷达创建一个dataset，并使用gzip压缩
            if radar_data_dict['RADAR_FRONT'] is not None:
                hf.create_dataset('radar_front', data=radar_data_dict['RADAR_FRONT'],
                                compression='gzip', compression_opts=9, chunks=True)
            if radar_data_dict['RADAR_FRONT_LEFT'] is not None:
                hf.create_dataset('radar_front_left', data=radar_data_dict['RADAR_FRONT_LEFT'],
                                compression='gzip', compression_opts=9, chunks=True)
            if radar_data_dict['RADAR_FRONT_RIGHT'] is not None:
                hf.create_dataset('radar_front_right', data=radar_data_dict['RADAR_FRONT_RIGHT'],
                                compression='gzip', compression_opts=9, chunks=True)
            if radar_data_dict['RADAR_BACK_LEFT'] is not None:
                hf.create_dataset('radar_back_left', data=radar_data_dict['RADAR_BACK_LEFT'],
                                compression='gzip', compression_opts=9, chunks=True)
            if radar_data_dict['RADAR_BACK_RIGHT'] is not None:
                hf.create_dataset('radar_back_right', data=radar_data_dict['RADAR_BACK_RIGHT'],
                                compression='gzip', compression_opts=9, chunks=True)

        # Save LIDAR data
        if lidar_data is not None:
            lidar_filename = self.lidar_path / f'{frame_str}.las'
            header = laspy.LasHeader(point_format=3, version="1.2")
            las = laspy.LasData(header)
            las.x = lidar_data[:,0]
            las.y = lidar_data[:,1]
            las.z = lidar_data[:,2]
            las.write(str(lidar_filename))
        
        self.count += 1

    def cleanup(self):
        """
        Clean up all sensors.
        """
        for sensor in self.sensors.values():
            if sensor is not None:
                sensor.destroy()


# ==============================================================================
# -- Main Simulation Loop -----------------------------------------------------
# ==============================================================================
#set_weather_by_preset用来选项式定义天气，总共有十四种天气可以定义
def set_weather_by_preset(sim_world, preset_name):
    if preset_name == 'ClearNoon':
        sim_world.set_weather(carla.WeatherParameters.ClearNoon)
    elif preset_name == 'CloudyNoon':
        sim_world.set_weather(carla.WeatherParameters.CloudyNoon)
    elif preset_name == 'WetNoon':
        sim_world.set_weather(carla.WeatherParameters.WetNoon)
    elif preset_name == 'WetCloudyNoon':
        sim_world.set_weather(carla.WeatherParameters.WetCloudyNoon)
    elif preset_name == 'MidRainyNoon':
        sim_world.set_weather(carla.WeatherParameters.MidRainyNoon)
    elif preset_name == 'HardRainNoon':
        sim_world.set_weather(carla.WeatherParameters.HardRainNoon)
    elif preset_name == 'SoftRainNoon':
        sim_world.set_weather(carla.WeatherParameters.SoftRainNoon)
    elif preset_name == 'ClearSunset':
        sim_world.set_weather(carla.WeatherParameters.ClearSunset)
    elif preset_name == 'CloudySunset':
        sim_world.set_weather(carla.WeatherParameters.CloudySunset)
    elif preset_name == 'WetSunset':
        sim_world.set_weather(carla.WeatherParameters.WetSunset)
    elif preset_name == 'WetCloudySunset':
        sim_world.set_weather(carla.WeatherParameters.WetCloudySunset)
    elif preset_name == 'MidRainSunset':
        sim_world.set_weather(carla.WeatherParameters.MidRainSunset)
    elif preset_name == 'HardRainSunset':
        sim_world.set_weather(carla.WeatherParameters.HardRainSunset)
    elif preset_name == 'SoftRainSunset':
        sim_world.set_weather(carla.WeatherParameters.SoftRainSunset)
    else:
        # 如果没有匹配上，就使用默认天气
        sim_world.set_weather(carla.WeatherParameters.ClearNoon)


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    env_manager = None
    player = None
    traffic_manager = None
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)

        # 如果用户指定了map，则加载指定的地图，否则使用当前的world
        if args.map is not None:
            sim_world = client.load_world(args.map)
            logging.info(f"Loaded world: {args.map}")
        else:
            sim_world = client.get_world()
            logging.info("Using current server world")

        set_weather_by_preset(sim_world, args.weather_preset)

        '''# 在此处添加修改交通灯时长的代码
        traffic_lights = sim_world.get_actors().filter('*traffic_light*')
        for tl in traffic_lights:
            tl.set_green_time(20.0)    # 绿灯20秒
            tl.set_yellow_time(3.0)    # 黄灯3秒
            tl.set_red_time(1.0)       # 红灯20秒
            # 可选：如果需要重置当前灯为绿色，并让它从绿灯开始倒计时
            tl.set_state(carla.TrafficLightState.Green)
            tl.reset_group()  # 使这个路口（group）立即开始新的周期
        sim_world.tick()'''

        # Set synchronous mode if specified
        if args.sync:
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05  # 20Hz
            sim_world.apply_settings(settings)
            logging.info("Synchronous mode enabled")

            # Set up Traffic Manager
            traffic_manager = client.get_trafficmanager(8000)
            traffic_manager.set_synchronous_mode(True)
            traffic_manager.set_global_distance_to_leading_vehicle(2.0)
            logging.info("Traffic Manager initialized in synchronous mode")

        # Get blueprint library
        blueprint_library = sim_world.get_blueprint_library()

        # Get vehicle blueprints
        vehicle_blueprints = blueprint_library.filter(args.filter)
        logging.info(f"Available vehicle blueprints: {[bp.id for bp in vehicle_blueprints]}")
        # 创建多个车辆
        carla_map = sim_world.get_map()
        spawn_points = carla_map.get_spawn_points()
        for i in range(100):
            cartype = random.randint(0, 20)
            vehicle_bp = blueprint_library.filter('vehicle')[cartype]  # 选择一种车辆类型
            spawn_point = spawn_points[i % len(spawn_points)]  # 循环选择不同的出生点
            vehicle = sim_world.spawn_actor(vehicle_bp, spawn_point)
            vehicle.set_autopilot(True)

        # Select the specified vehicle blueprint
        selected_bp = None
        for bp in vehicle_blueprints:
            if args.filter in bp.id:
                selected_bp = bp
                break

        if selected_bp is None:
            logging.warning(f"Blueprint {args.filter} not found. Using default vehicle.")
            selected_bp = vehicle_blueprints[0]
            logging.info(f"Using '{selected_bp.id}' as Ego vehicle.")
        else:
            logging.info(f"Using '{selected_bp.id}' as Ego vehicle.")

        # Set ego vehicle attributes
        selected_bp.set_attribute('role_name', 'hero')

        # Get spawn points
        spawn_points = sim_world.get_map().get_spawn_points()
        if not spawn_points:
            logging.error("No spawn points available")
            return

        # Attempt to spawn the ego vehicle
        player = None
        for spawn_point in spawn_points:
            player = sim_world.try_spawn_actor(selected_bp, spawn_point)
            if player:
                logging.info(f"Successfully spawned Ego vehicle at {spawn_point.location}")
                break

        if player is None:
            logging.error("Ego vehicle could not be spawned at any spawn point")
            return

        # Enable autopilot
        player.set_autopilot(True, traffic_manager.get_port() if traffic_manager else 0)
        logging.info("Autopilot enabled for Ego vehicle.")

        # Add ego vehicle to Traffic Manager
        if traffic_manager:
            traffic_manager.vehicle_percentage_speed_difference(player, 0.0)  # Ego maintains its speed
            # 让车辆完全无视红绿灯
            traffic_manager.ignore_lights_percentage(player, 100)

        # Initialize environment manager
        env_manager = Env_Manager(sim_world, player, save_path='scenario_name')

        # Initialize Pygame display window (optional)
        display_width, display_height = args.width, args.height
        display = pygame.display.set_mode((display_width, display_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("CARLA Data Collection")

        clock = pygame.time.Clock()
        max_frames = args.max_frames  # Maximum frames to collect

        frame_counter = 0

        while True:
            clock.tick_busy_loop(60)  # Control frame rate

            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt()

            # Advance the simulation
            if args.sync:
                sim_world.tick()
            else:
                sim_world.wait_for_tick()

            # Process sensor data
            results = env_manager.tick()
            if results:
                # Save current frame's data
                frame_counter += 1
                if frame_counter % save_interval == 0:
                    env_manager.save(results)
                    logging.info(f"Data saved at frame {frame_counter}")

                # Print vehicle's location and speed
                annotation = results['annotation']
                location = annotation['x'], annotation['y'], annotation['theta']
                speed = annotation['speed']
                print(f"Frame {frame_counter}: Location - x: {annotation['x']:.2f}, y: {annotation['y']:.2f}, theta: {annotation['theta']:.2f}; Speed: {speed:.2f} km/h")

                # 渲染前置RGB相机图像到Pygame窗口
                cam_front = results['camera_data'].get('rgb_front', None)
                if cam_front is not None:
                    if cam_front.dtype != np.uint8:
                        cam_front = np.clip(cam_front, 0, 255).astype(np.uint8)
                    if cam_front.shape[2] != 3:
                        print(f"Frame {frame_counter}: Unexpected number of channels in CAM_FRONT: {cam_front.shape[2]}")
                        continue
                    img = cv2.resize(cam_front, (display_width, display_height))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    try:
                        surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
                        display.blit(surface, (0, 0))
                        pygame.display.flip()
                    except Exception as e:
                        print(f"Frame {frame_counter}: Error rendering image: {e}")
                else:
                    print(f"Frame {frame_counter}: No CAM_FRONT data to render.")

            # Check if maximum frame count is reached
            if frame_counter >= max_frames:
                logging.info("Reached maximum frame count. Exiting simulation.")
                break

        logging.info("Data collection completed")

    finally:
        if env_manager is not None:
            env_manager.cleanup()
        if traffic_manager is not None:
            traffic_manager.set_synchronous_mode(False)
        if world is not None:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        if player is not None:
            player.destroy()

        pygame.quit()

def main():
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='CARLA Data Collection Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--map',
        default=None,
        help='Map name to load (e.g. Town01, Town02, etc.)')
    argparser.add_argument(
        '--weather_preset',
        default='ClearNoon',
        choices=['ClearNoon', 'CloudyNoon', 'WetNoon', 'WetCloudyNoon',
                 'MidRainyNoon', 'HardRainNoon', 'SoftRainNoon',
                 'ClearSunset', 'CloudySunset', 'WetSunset', 'WetCloudySunset',
                 'MidRainSunset', 'HardRainSunset', 'SoftRainSunset'],
        help='Choose a weather preset for the simulation')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1600x900',
        help='Window resolution (default: 1600x900)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.tesla.model3',
        help='Actor filter (default: "vehicle.tesla.model3")')
    argparser.add_argument(
        '-m', '--max_frames',
        metavar='N',
        default=2000,
        type=int,
        help='Maximum number of frames to collect (default: 2000)')
    argparser.add_argument(
        '-a', '--agent', type=str,
        choices=["Behavior", "Basic", "Constant"],
        help="Select which agent to run",
        default="Behavior")
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal)',
        default='normal')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('Listening to server %s:%s', args.host, args.port)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        logging.info('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
