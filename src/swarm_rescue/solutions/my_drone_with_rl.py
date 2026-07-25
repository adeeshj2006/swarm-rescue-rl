from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.utils.misc_data import MiscData
from swarm_rescue.simulation.utils.utils import normalize_angle
import numpy as np
import random

class MyDroneRandom(DroneAbstract):
    def __init__(self, identifier = None, misc_data = None, display_lidar_graph = False, **kwargs):
        super().__init__(identifier, misc_data, display_lidar_graph, **kwargs)
        self.pending_action = {"forward":0, "lateral": 0, "rotation": 0, "grasper": 0}

    def control(self):
        return self.pending_action

    def define_message_for_all(self):
        return None

    def brain_nn(self):
        id = self.identifier            # identifier: int
        mass = self._base._mass         # _mass: float | None
        map_size = self.size_area       # Optional[tuple]: The size of the area.
        lidar = self.lidar_values()     # Any: Sensor values. | fov 360deg | resolution 181 | range 300px
        lidar /= 300
        lidar_is_disabled = self.lidar_is_disabled()    # bool: True if disabled, False otherwise.
        semantic = self.semantic_values()       # Any: Sensor values | 360 deg fov | resolution 35 | range 200px
        semantic_is_disabled = self.semantic_is_disabled()  # bool: True if disabled, False otherwise.
        gps = self.measured_gps_position()      # Get the measured position of the drone, in pixels. | Returns: Union[np.ndarray, None]: The measured position, or None if unavailable.
        gps[0] /= map_size[0]
        gps[1] /= map_size[1]
        gps_is_disabled = self.gps_is_disabled()    # bool: True if disabled, False otherwise.
        compass = self.measured_compass_angle()     # Get the measured orientation of the drone, in radians between -Pi and Pi. | Returns: Union[float, None]: The measured angle, or None if unavailable.
        compass /= np.pi
        compass_is_disabled = self.compass_is_disabled()    # bool: True if disabled, False otherwise.
        odom = self.odometer_values()
        #DroneOdometer sensor returns a numpy array containing:
            #   - dist_travel: the distance of the travel of the drone during one step
            #   - alpha: the relative angle of the current position seen from the
            #     previous reference frame of the drone
            #   - theta: the variation of orientation (or rotation) of the drone during
            #     the last step in the reference frame
        
        odometer_is_disabled = self.odometer_is_disabled()  # bool: True if disabled, False otherwise.
        health = self.drone_health      # int: The drone's health.
        is_inside_return_area = self.is_inside_return_area      # is_inside_return_area: bool
        elapsed_timestep = self.elapsed_timestep        # elapsed_timestep: int
        elapsed_walltime = self.elapsed_walltime        # elapsed_walltime: int
        grasped_wounded_persons = self.grasped_wounded_persons()        # list: List of grasped wounded persons.
        measured_velocity = self.measured_velocity() / 4        # Returns: Union[np.ndarray, None]: The measured velocity, or None if unavailable.
        measured_angular_velocity = self.measured_angular_velocity() / 0.18   # Returns: Union[float, None]: The measured angular velocity, or None if unavailable.
        rand = random.random()      # Random

        obs = []
        obs.extend(lidar)
        obs.extend(semantic.flatten())
        obs.extend(gps if not gps_is_disabled else [0, 0])
        obs.append(compass if not compass_is_disabled else 0)
        obs.extend(odom)
        obs.extend(measured_velocity if measured_velocity is not None else [0, 0])
        obs.append(measured_angular_velocity if measured_angular_velocity is not None else 0)
        obs.append(health)
        obs.append(float(is_inside_return_area))
        obs.append(float(len(grasped_wounded_persons) > 0))
