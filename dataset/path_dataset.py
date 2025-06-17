import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm

import random

from config import *
from dataset.astar import AStar, generate_obstacle_map, apply_buffer, plot_obstacle_map, generate_random_non_colliding_point, resample_path_to_length

OBSTACLE_MAP_SIZE = (20, 20)

def generate_synthetic_data(num_samples, map_size, n_waypoints, path_dim):
    
    obstacle_maps_list = []
    start_positions_list = []
    goal_positions_list = []
    expert_paths_list = []

    '''
    for i in num_samples:
        for j in num_attempts:
            generate an obstacle map
            randomly initialise a start position
            randomly initialise a goal position

            generate path

            if path is not possible:
                continue
                                
            obstacle_maps_list.append(omap)
            expert_paths_list.append(start_pos)
            expert_paths_list.append(goal_position)
            expert_paths_list.append(path)
    '''

    num_attempts = 20
    num_obstacles = 10 # later change it to random int in a range


    for i in range(num_samples):
        for j in range(num_attempts):
            size = (20, 20)
            num_obstacles = 10
            buffer_radius = 1

            original_map = generate_obstacle_map(size, num_obstacles)
            buffered_map = apply_buffer(original_map, buffer_radius)

            start_tuple = generate_random_non_colliding_point(buffered_map)
            

            free_cells_row_indices, free_cells_col_indices = np.nonzero(buffered_map == 0)
            all_free_cells = list(zip(free_cells_row_indices, free_cells_col_indices))
            
            available_goal_cells = [cell for cell in all_free_cells if cell != start_tuple]
            
            if not available_goal_cells:
                print("Warning: Only one free cell available. Start and Goal will be the same.")
                goal_tuple = start_tuple
            else:
                goal_tuple = random.choice(available_goal_cells)


            start = (float(start_tuple[0]), float(start_tuple[1]))
            goal = (float(goal_tuple[0]), float(goal_tuple[1]))

            astar = AStar(buffered_map, start, goal, step_size=1.0)
            path = astar.generate_path()

            if path:
                path = resample_path_to_length(path, N_WAYPOINTS)
                obstacle_maps_list.append(original_map) # not the buffer map, maybe it would learn to keep the buffer by itself - let's see
                start_positions_list.append(start)
                goal_positions_list.append(goal)
                expert_paths_list.append(path)

                break
            else:
                continue


    print("Generated {} samples with obstacle maps, start positions, goal positions, and expert paths.".format(num_samples))


    return (torch.tensor(np.array(obstacle_maps_list), dtype=torch.float32),
                torch.tensor(np.array(start_positions_list), dtype=torch.float32),
                torch.tensor(np.array(goal_positions_list), dtype=torch.float32),
                torch.tensor(np.array(expert_paths_list), dtype=torch.float32))

class PathDataset(Dataset):
    def __init__(self, num_samples, map_size, n_waypoints, path_dim):
        self.obstacle_maps, self.start_pos, self.goal_pos, self.expert_paths = \
            generate_synthetic_data(num_samples, map_size, n_waypoints, path_dim)

    def __len__(self):
        return len(self.expert_paths)

    def __getitem__(self, idx):
        # Get the 2D obstacle map
        obstacle_map_2d = self.obstacle_maps[idx]  # Shape: (20, 20)

        # Add a channel dimension: (H, W) -> (1, H, W)
        # This is crucial for Conv2d layers
        obstacle_map_3d = obstacle_map_2d.unsqueeze(0) # Shape: (1, 20, 20)

        # Normalize path coordinates to [-1, 1] if map is [0, map_size]
        map_max_dim = max(OBSTACLE_MAP_SIZE) # OBSTACLE_MAP_SIZE is (20,20) from this file or imported
        expert_path_norm = (self.expert_paths[idx] / map_max_dim) * 2 - 1
        start_pos_norm = (self.start_pos[idx] / map_max_dim) * 2 - 1
        goal_pos_norm = (self.goal_pos[idx] / map_max_dim) * 2 - 1

        return (obstacle_map_3d, start_pos_norm, # Return the map with channel dim
                goal_pos_norm, expert_path_norm)
