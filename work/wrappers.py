import gymnasium as gym
from gymnasium import spaces
import numpy as np

from sklearn.neighbors import KDTree

class LidarRandomizer(gym.ObservationWrapper):
    def __init__(self, env, epsilon=0.02, zone_p=0.01, extreme_p=0.005):
        super().__init__(env)
        self.epsilon = epsilon
        self.zone_p = zone_p
        self.extreme_p = extreme_p

    def observation(self, obs):
        lidar_data = obs["scans"]
        
        # Try normal vs uniform noise
        noise = np.random.uniform(-self.epsilon, self.epsilon, size=lidar_data.shape)
        lidar_data += noise

        # Randomly choose areas to increase/decrease.
        if np.random.random() < self.zone_p:
            # Define size of the area (20% of the readings).
            size = int(len(lidar_data) * 0.2)
            start = np.random.randint(0, len(lidar_data) - size)
            end = start + size
            # Randomly choose whether to increase or decrease, and by how much.
            change = np.random.uniform(-0.1, 0.1)
            lidar_data[start:end] += change

        # Randomly set some readings to very high or very low.
        if np.random.random() < self.extreme_p:
            index = np.random.randint(len(lidar_data))
            lidar_data[index] = np.random.choice([0.1, 0.9])

        # Make sure the output is still between 0 and 1.
        lidar_data = np.clip(lidar_data, 0, 1)
        
        
        obs["scans"] = lidar_data

        return obs
    

class ActionRandomizer(gym.ActionWrapper):
    def __init__(self, env, epsilon=0.05):
        super().__init__(env)
        self.epsilon = epsilon
    def action(self, action):
        noise = np.random.uniform(-self.epsilon, self.epsilon, size=action.shape)
        action = np.clip(action + noise, self.action_space.low, self.action_space.high)
        return action


class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, obs):
        vs = obs["linear_vels_s"][self.ego_idx]
        vd = obs["linear_vels_d"][self.ego_idx]
        d = obs["poses_d"]

        reward = 0.0
        
        if abs(obs["linear_vels_x"]) <= 0.2:
            reward -= 5.0
        
        # Encourage the agent to move in the vs direction
        reward += 1.0 * vs
        reward -= 0.01 * abs(vd)
                
        reward -= 0.02 * abs(d)
        reward -= 0.1 * abs(self.action[0])        

        # Penalize the agent for collisions
        if self.env.collisions[0]:
            reward -= 1000.0
            
        return reward

    def step(self, action):
        self.action = action
        obs, _, terminated, truncated, info = self.env.step(action)
        return obs, self.reward(obs).item(), terminated, truncated, info


class FrenetObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(FrenetObsWrapper, self).__init__(env)

        self.map_data = env.map_data.to_numpy()
        self.kdtree = KDTree(self.map_data[:, 1:3])
        
        na = self.num_agents
        self.observation_space = spaces.Dict({
            "scans": spaces.Box(0, 1, (env.num_beams,), np.float32),
            "poses_s":       spaces.Box(-100, 100, (1,), np.float32),
            "poses_d":       spaces.Box(-100, 100, (1,), np.float32),
            "linear_vels_s": spaces.Box(-10, 10, (1,), np.float32),
            "linear_vels_d": spaces.Box(-10, 10, (1,), np.float32),
            "linear_vel":    spaces.Box(0, 1, (na,), np.float32),
            "prev_action":   spaces.Box(np.array([-0.4189, 0.01]), 
                                        np.array([0.4189, 3.2 ]), dtype=np.float64), 
            })

    def observation(self, obs):
        new_obs = obs.copy()
        # print(self.env.prev_action)
        
        theta = new_obs["poses_theta"]
        scans = new_obs["scans"]
        px = new_obs["poses_x"]
        py = new_obs["poses_y"]
        vx = new_obs["linear_vels_x"]
        vy = new_obs["linear_vels_y"]
        v = (vx**2 + vy**2)**0.5

        frenet_coords = to_frenet(px[0], py[0], v, theta[0], self.map_data, self.kdtree)
        
        new_obs["poses_s"] = frenet_coords[0]
        new_obs["poses_d"] = frenet_coords[1]
        new_obs["linear_vels_s"] = frenet_coords[2]
        new_obs["linear_vels_d"] = frenet_coords[3]
        
        # Save the first lidar data
        # np.save('lidar_data_sim', np.array(new_obs["scans"]))
        
        # Preprocess the scans and add linear_vel
        mask = scans >= 10
        noise = np.random.uniform(-0.5, 0, mask.sum())
        scans = np.clip(scans, None, 10)
        scans[mask] += noise
        scans /= 10.0
        
        new_obs["scans"] = scans
        new_obs["linear_vel"] = v / 3.2
        new_obs["prev_action"] = self.env.prev_action
        
        return new_obs

def get_closest_point_index(x, y, kdtree):
    _, indices = kdtree.query(np.array([[x, y]]), k=1)
    closest_point_index = indices[0, 0]
    return closest_point_index

def to_frenet(x, y, vel_magnitude, pose_theta, map_data, kdtree):
    closest_point_index = get_closest_point_index(x, y, kdtree)
    closest_point = map_data[closest_point_index]
    s_m, x_m, y_m, psi_rad = closest_point[0:4]

    dx = x - x_m
    dy = y - y_m

    s = -dx * np.sin(psi_rad) + dy * np.cos(psi_rad) + s_m
    d = dx * np.cos(psi_rad) + dy * np.sin(psi_rad)

    vs = vel_magnitude * np.sin(pose_theta - psi_rad)
    vd = vel_magnitude * np.cos(pose_theta - psi_rad)

    return s, d, vs, vd