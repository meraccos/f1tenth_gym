# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import random
import time

import cv2
import numpy as np
import yaml
from f110_gym.envs.base_classes import Integrator, Simulator
from stable_baselines3.common.logger import read_csv

import gymnasium as gym
from gymnasium import spaces

# Constants
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800
DTYPE = np.float64


class F110Env(gym.Env):
    """
    OpenAI gym environment for F1TENTH

    Env should be initialized by calling gym.make('f110_gym:f110-v0', **kwargs)

    Args:
        kwargs:
            seed (int, default=12345): seed for random state and reproducibility

            map (str, default='vegas'): name of the map used for the environment. Currently, available environments include: 'berlin', 'vegas', 'skirk'. You could use a string of the absolute path to the yaml file of your custom map.

            map_ext (str, default='png'): image extension of the map image file. For example 'png', 'pgm'

            params (dict, default={'mu': 1.0489, 'C_Sf':, 'C_Sr':, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch':7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}): dictionary of vehicle parameters.
            mu: surface friction coefficient
            C_Sf: Cornering stiffness coefficient, front
            C_Sr: Cornering stiffness coefficient, rear
            lf: Distance from center of gravity to front axle
            lr: Distance from center of gravity to rear axle
            h: Height of center of gravity
            m: Total mass of the vehicle
            I: Moment of inertial of the entire vehicle about the z axis
            s_min: Minimum steering angle constraint
            s_max: Maximum steering angle constraint
            sv_min: Minimum steering velocity constraint
            sv_max: Maximum steering velocity constraint
            v_switch: Switching velocity (velocity at which the acceleration
                is no longer able to create wheel spin)
            a_max: Maximum longitudinal acceleration
            v_min: Minimum longitudinal velocity
            v_max: Maximum longitudinal velocity
            width: width of the vehicle in meters
            length: length of the vehicle in meters

            num_agents (int, default=2): number of agents in the environment

            timestep (float, default=0.01): physics timestep

            ego_idx (int, default=0): ego's index in list of agents
    """
    
    # rendering
    renderer = None
    render_callbacks = []

    def __init__(self, **kwargs):
        # kwargs extraction     
        self.seed = kwargs.get('seed', 12345)
        
        self.maps = kwargs.get('maps')
        
        default_params = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 
                          'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 
                          'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min':-5.0, 
                          'v_max': 20.0, 'width': 0.31, 'length': 0.58}
        self.params = kwargs.get('params', default_params)
        
        # simulation parameters
        self.num_agents = kwargs.get('num_agents', 2)
        self.timestep = kwargs.get('timestep', 0.01)
        self.ego_idx = kwargs.get('ego_idx', 0)
        
        self.integrator = kwargs.get('integrator', Integrator.RK4)
           
        # number of lidar beams
        self.num_beams = kwargs.get('num_beams', 2155)

        # env states
        self.poses = np.zeros((self.num_agents, 3))
        self.collisions = np.zeros(self.num_agents)
        self.prev_action = np.array([0.0,0.01])

        # race info
        self.lap_times = np.zeros(self.num_agents)
        self.lap_counts = np.zeros(self.num_agents)

        # initiate stuff
        self.sim = Simulator(self.params, self.num_agents, 
                             seed=self.seed, num_beams=self.num_beams, 
                             time_step=self.timestep,
                             integrator=self.integrator)
        self._set_random_map()
        
        # stateful observations for rendering
        self.render_obs = None
        
        self.metadata = {'render_modes': ['human', 'human_fast'], 'render_fps': 60}
        
        self.action_space = spaces.Box(np.array([self.params['s_min'], 
                                                 0.01]), 
                                       np.array([self.params['s_max'],
                                                 3.2 ]), 
                                       dtype=np.float64)
        
        self.observation_space = spaces.Dict({
            'scans': spaces.Box(0, 100, (self.num_beams, ), DTYPE),
            'poses_x': spaces.Box(-1000, 1000, (self.num_agents,), DTYPE),
            'poses_y': spaces.Box(-1000, 1000, (self.num_agents,), DTYPE),
            'poses_theta': spaces.Box(-2*np.pi, 2*np.pi, (self.num_agents,),DTYPE),
            'linear_vels_x': spaces.Box(-10, 10, (self.num_agents,), DTYPE),
            'linear_vels_y': spaces.Box(-10, 10, (self.num_agents,), DTYPE),
            'ang_vels_z': spaces.Box(-10, 10, (self.num_agents,), DTYPE),
            'collisions': spaces.Box(0, 1, (self.num_agents,), DTYPE),
            'lap_times': spaces.Box(0, 1e6, (self.num_agents,), DTYPE),
            'lap_counts': spaces.Box(0, 9, (self.num_agents,), np.int32)
        })

    def find_closest_index(self, sorted_list, target):
        left, right = 0, len(sorted_list) - 1

        while left < right:
            mid = (left + right) // 2

            if sorted_list[mid] == target:
                return mid
            elif sorted_list[mid] < target:
                left = mid + 1
            else:
                right = mid

        cond = abs(sorted_list[left - 1] - target) <= abs(sorted_list[left] - target)
        if left > 0 and cond:
            return left - 1

        return left

    def add_random_shapes(self, image_path, coordinates_list):
        image = cv2.imread(image_path)
        shape_choices = ["rectangle", "circle", "triangle",
                         "ellipse", "rounded_rectangle",
                         "pentagon", "hexagon", "star"]

        for coord in coordinates_list:
            x = int(coord[0])
            y = 1600 - int(coord[1])
            size = int(coord[2])
            shape = random.choice(shape_choices)
            color = (0, 0, 0)

            if shape == "rectangle":
                center_x, center_y = x + size//2, y + size//2
                cv2.rectangle(image, (center_x - size//2, center_y - size//2),
                              (center_x + size//2, center_y + size//2),
                              color, -1)

            elif shape == "circle":
                cv2.circle(image, (x, y), size // 2, color, -1)

            elif shape == "triangle":
                points = np.array([[(x + size//2, y),
                                    (x + size, y + size),
                                    (x, y + size)]], dtype=np.int32)
                cv2.fillPoly(image, points, color)

            elif shape == "ellipse":
                center_x, center_y = x + size//2, y + size//2
                cv2.ellipse(image,
                            (center_x, center_y),
                            (size//2, size//4),
                            0, 0, 360, color, -1)

            elif shape == "rounded_rectangle":
                rx, ry = size // 5, size // 5
                cv2.rectangle(image, (x + rx, y), (x + size - rx, y + size), color, -1)
                cv2.rectangle(image, (x, y + ry), (x + size, y + size - ry), color, -1)
                cv2.circle(image, (x + rx, y + ry), ry, color, -1)
                cv2.circle(image, (x + size - rx, y + ry), ry, color, -1)
                cv2.circle(image, (x + rx, y + size - ry), ry, color, -1)
                cv2.circle(image, (x + size - rx, y + size - ry), ry, color, -1)
                
            elif shape == "pentagon":
                points = np.array([[x + size // 2, y],
                                   [x + size, y + size // 3],
                                   [x + 3 * size // 4, y + size],
                                   [x + size // 4, y + size],
                                   [x, y + size // 3]], dtype=np.int32)
                cv2.fillPoly(image, [points], color)
                
            elif shape == "hexagon":
                points = np.array([[x + size // 4, y],
                                   [x + 3 * size // 4, y],
                                   [x + size, y + size // 2],
                                   [x + 3 * size // 4, y + size],
                                   [x + size // 4, y + size],
                                   [x, y + size // 2]], dtype=np.int32)
                cv2.fillPoly(image, [points], color)
                
            elif shape == "star":
                outer_radius = size // 2
                inner_radius = outer_radius // 2
                angle = 2 * np.pi / 10
                points = []
                for i in range(10):
                    if i % 2 == 0:
                        r = outer_radius
                    else:
                        r = inner_radius
                    x_offset = x + r * np.cos(i * angle)
                    y_offset = y + r * np.sin(i * angle)
                    points.append([x_offset, y_offset])
                points = np.array(points, dtype=np.int32)
                cv2.fillPoly(image, [points], color)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Save the edited image
        file_name, file_extension = os.path.splitext(image_path)
        output_image_path = file_name + "_obs" + file_extension
        
        self.map_name = self.map_name + "_obs"
        self.map_png = f"{self.map_dir}/maps/{self.map_name}.png"
        self.map_yaml = f"{self.map_dir}/maps/{self.map_name}.yaml"
        
        cv2.imwrite(output_image_path, image)

    def add_obstacles(self):
        s_data = self.map_data["s_m"].to_numpy()
        num_obstacles = random.randint(3, 20)
        ds = self.map_max_s / num_obstacles
        obs_data = []

        for i in range(1, num_obstacles):
            target = i * ds
            closest_index = self.find_closest_index(s_data, target)
            width = self.map_width

            obs_size = width / random.uniform(1.0, 4.0)
            obs_x = self.map_data["x_m"].iloc[closest_index]
            obs_y = self.map_data["y_m"].iloc[closest_index]

            random_disp = random.uniform(0, width)
            random_angle = random.uniform(0, 2*np.pi)

            obs_x_img = (obs_x - self.map_origin[0]) / self.map_resolution
            obs_y_img = (obs_y - self.map_origin[1]) / self.map_resolution

            random_dx = random_disp * np.cos(random_angle)
            random_dy = random_disp * np.sin(random_angle)

            obs_x_img += random_dx
            obs_y_img += random_dy

            obs_data.append((obs_x_img, obs_y_img, obs_size))

        self.add_random_shapes(image_path=self.map_png, coordinates_list=obs_data)

    def _set_random_map(self):
        self.map_idx = random.choice(self.maps)
        self.map_dir = '/Users/meraj/workspace/f1tenth_gym/work/tracks'
        self.map_name = 'map{}'.format(self.map_idx)

        self.map_yaml = f"{self.map_dir}/maps/{self.map_name}.yaml"
        self.map_png = f"{self.map_dir}/maps/{self.map_name}.png"
        self.map_csv = f"{self.map_dir}/centerline/{self.map_name}.csv"
        
        self.map_data = read_csv(self.map_csv)
        self.map_width = self.map_data["width"].iloc[0] * 2.0

        with open(self.map_yaml, 'r') as file:
            yaml_data = yaml.safe_load(file)

        self.map_origin = yaml_data['origin'][0:2]
        self.map_resolution = yaml_data['resolution']
        self.map_max_s = self.map_data["s_m"].iloc[-1]

        self.add_obstacles()
        
        self.update_map(self.map_yaml)

    def update_map(self, map_path):
        """
        Updates the map used by the simulation.

        Args:
            map_path (str): Absolute path to the map YAML file.
            map_ext (str): Extension of the map image file.
        """
        self.sim.set_map(map_path, '.png')

    def __del__(self):
        """
        Finalizer, does cleanup
        """
        pass

    def _check_done(self):
        """
        Check if the current rollout is done

        Args:
            None

        Returns:
            done (bool): whether the rollout is done
            toggle_list (list[int]): each agent's toggle list
                        for crossing finish line
        """

        # this is assuming 2 agents
        # TODO: switch to maybe s-based
        left_t, right_t = 2, 2

        poses_x = self.poses_x-self.start_xs
        poses_y = self.poses_y-self.start_ys
        delta_pt = np.dot(self.start_rot, np.stack((poses_x, poses_y)))
        temp_y = delta_pt[1, :]
        temp_y = np.where(temp_y > left_t, temp_y - left_t,
                          np.where(temp_y < -right_t, -right_t - temp_y, 0))

        dist2 = delta_pt[0]**2 + temp_y**2
        closes = dist2 <= 0.1

        self.toggle_list = np.where(np.logical_xor(closes, self.near_starts),
                                    self.toggle_list + 1, self.toggle_list)
        self.near_starts = closes.copy()
        self.lap_counts = self.toggle_list // 2
        self.lap_times = np.where(self.toggle_list < 4,
                                  self.current_time,
                                  self.lap_times)

        done = (self.collisions[self.ego_idx]) or np.all(self.toggle_list >= 2)        
            
        return bool(done)


    def _update_state(self, obs_dict):
        for key in ['poses_x', 'poses_y', 'poses_theta', 'collisions']:
            setattr(self, key, obs_dict[key])

    def _format_obs(self, obs):
        formatted_obs = {
            key: np.array(value, dtype=DTYPE)
            for key, value in obs.items()}
        formatted_obs['lap_counts'] = np.array(obs['lap_counts'], np.int32)
        return formatted_obs

    def _update_render_obs(self, obs):
        self.render_obs = {
            key: obs[key] for key in ['poses_x', 'poses_y', 'poses_theta',
                                      'lap_times', 'lap_counts']
        }

    def step(self, action):
        # call simulation step
        self.prev_action = action
        obs = self.sim.step(action)
        obs['lap_times'] = self.lap_times
        obs['lap_counts'] = self.lap_counts
        self._update_render_obs(obs)
        self._update_state(obs)
        self.current_time += self.timestep

        # check done
        done = self._check_done()
        
        info = {'lap_count': self.lap_counts,
                'collision': self.collisions[0],
                'is_success': obs['lap_counts'][0] >=1}

        # Reverse the lidar data
        obs['scans'] = obs['scans'][0][::-1]
        obs = self._format_obs(obs)

        return obs, 0, done, False, info

    def reset(self, poses=None, seed=None, options=None):
        """
        Args:
            poses (np.ndarray (num_agents, 3)): poses to reset agents to

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """
        super().reset(seed=seed)
        self._set_random_map()

        if poses is None:
            # Generate random poses for the agents
            init_x = np.random.uniform(-0.3, 0.3)
            init_y = np.random.uniform(-0.3, 0.3)
            init_angle = (np.pi/2 +
                          self.map_data["psi_rad"].iloc[1] +
                          np.random.uniform(-np.pi/12, np.pi/12))
                        #   np.random.uniform(-np.pi/3, np.pi/3))

            poses = np.array([[init_x, init_y, init_angle]])

        # reset counters and data members
        self.current_time = 0.0
        self.collisions = np.zeros((self.num_agents, ))
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True]*self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))

        # states after reset
        self.start_xs = poses[:, 0]
        self.start_ys = poses[:, 1]
        self.start_thetas = poses[:, 2]
        self.start_rot = np.array([[np.cos(-self.start_thetas[self.ego_idx]),
                                    -np.sin(-self.start_thetas[self.ego_idx])],
                                   [np.sin(-self.start_thetas[self.ego_idx]),
                                    np.cos(-self.start_thetas[self.ego_idx])]])

        # call reset to simulator
        self.sim.reset(poses)

        # get no input observations
        action = np.zeros((self.num_agents, 2))
        obs, reward, terminated, truncated, info = self.step(action)

        self._update_render_obs(obs)
        obs = self._format_obs(obs)
        
        return obs, info

    def update_params(self, params, index=-1):
        """
        Updates the parameters used by the simulation for vehicles.

        Args:
            params (dict): Dictionary of parameters.
            index (int, default=-1): If >= 0 then only
                    update a specific agent's params.
        """
        self.sim.update_params(params, agent_idx=index)

    def add_render_callback(self, callback_func):
        """
        Add an extra drawing function to call during rendering.

        Args:
            callback_func (function (EnvRenderer) -> None): Custom function
                            to be called during render().
        """

        F110Env.render_callbacks.append(callback_func)

    def render(self, mode='human'):
        """
        Renders the environment with Pyglet. Use mouse scroll in the window to
        zoom in/out, use mouse click drag to pan. Shows the agents, the map,
        current FPS (bottom-left corner), and the race information as text.

        Args:
            mode (str, default='human'): Rendering mode, currently supports:
                'human': Slowed down rendering such that the env is
                        rendered in a way that sim time elapsed is
                        close to real time elapsed.
                'human_fast': Render as fast as possible.
        """
        assert mode in ['human', 'human_fast']

        if F110Env.renderer is None:
            # first call, initialize everything
            from f110_gym.envs.rendering import EnvRenderer
            F110Env.renderer = EnvRenderer(WINDOW_W, WINDOW_H)
            png_dir = f"{self.map_dir}/maps/{self.map_name}"
            F110Env.renderer.update_map(png_dir, '.png')

        F110Env.renderer.update_obs(self.render_obs)

        for render_callback in F110Env.render_callbacks:
            render_callback(F110Env.renderer)

        F110Env.renderer.dispatch_events()
        F110Env.renderer.on_draw()
        F110Env.renderer.flip()
        if mode == 'human':
            time.sleep(0.005)
        elif mode == 'human_fast':
            pass
