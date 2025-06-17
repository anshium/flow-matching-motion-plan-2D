import numpy as np
import matplotlib.pyplot as plt
import heapq
from scipy.ndimage import binary_dilation
from scipy.interpolate import interp1d
import random

class AStar:
    def __init__(self, obstacle_map, start, goal, step_size=1.0):
        self.obstacle_map = obstacle_map
        self.start = start
        self.goal = goal
        self.step_size = step_size
        self.path = []

    def is_valid(self, point):
        """
        Checks if a point is within map bounds and not in an obstacle.
        """
        x, y = point
        if not (0 <= x < self.obstacle_map.shape[0] and 0 <= y < self.obstacle_map.shape[1]):
            return False
        
        return self.obstacle_map[int(x), int(y)] == 0

    def heuristic(self, a, b):
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5

    def generate_neighbors(self, current):
        directions = [
            (1, 0), (0, 1), (-1, 0), (0, -1), 
            (1, 1), (-1, 1), (-1, -1), (1, -1) 
        ]
        neighbors = []
        for dx, dy in directions:
            new_point = (current[0] + dx * self.step_size, current[1] + dy * self.step_size)
            if self.is_valid(new_point):
                neighbors.append(new_point)
        return neighbors

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    def generate_path(self):
        open_set = []
        heapq.heappush(open_set, (self.heuristic(self.start, self.goal), 0, self.start))

        g_score = {self.start: 0}
        
        came_from = {}
    
        def get_point_key(p):
            return (int(round(p[0])), int(round(p[1])))

        visited_keys = set()

        while open_set:
            f_score, current_g_score, current = heapq.heappop(open_set)
            
        
            if self.heuristic(current, self.goal) < self.step_size:
                self.path = self.reconstruct_path(came_from, current)
                return self.path

            current_key = get_point_key(current)
            if current_key in visited_keys:
                continue
            visited_keys.add(current_key)

            for neighbor in self.generate_neighbors(current):
                tentative_g_score = current_g_score + self.step_size

            
                neighbor_key = get_point_key(neighbor)

                if neighbor_key not in g_score or tentative_g_score < g_score[neighbor_key]:
                    g_score[neighbor_key] = tentative_g_score
                    came_from[neighbor] = current
                    f_score = tentative_g_score + self.heuristic(neighbor, self.goal)
                    heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))
        
        return None

def generate_obstacle_map(size, num_obstacles):
    obstacle_map = np.zeros(size)
    for _ in range(num_obstacles):
        x, y = np.random.randint(0, size[0]), np.random.randint(0, size[1])
        obstacle_map[x, y] = 1 
    return obstacle_map

def apply_buffer(obstacle_map, buffer_radius=1):
    structure = np.ones((2 * buffer_radius + 1, 2 * buffer_radius + 1))
    buffered_map = binary_dilation(obstacle_map, structure=structure).astype(int)
    return buffered_map

def plot_obstacle_map(original_map, buffered_map=None, path=None):
    combined = np.zeros_like(original_map)

    original_map_bool = original_map.astype(bool)
    buffered_map_bool = buffered_map.astype(bool)

    combined[buffered_map_bool] = 0.5 
    combined[original_map_bool] = 1.0 


    plt.imshow(combined, cmap='gray', origin='lower')
    plt.colorbar(label='0: Free, 0.5: Buffer, 1: Obstacle')

    if path:
        path = np.array(path)
    
        plt.plot(path[:, 1], path[:, 0], color='red', linewidth=2, label='A* Path')
    
        plt.scatter(path[0, 1], path[0, 0], color='green', marker='o', s=100, label='Start')
        plt.scatter(path[-1, 1], path[-1, 0], color='blue', marker='o', s=100, label='Goal')

    plt.legend()
    plt.title('Obstacle Map with Buffer and A* Path')
    plt.grid(True, which='both', color='lightgray', linestyle='-', linewidth=0.5)
    plt.xticks(np.arange(-0.5, original_map.shape[1], 1), [])
    plt.yticks(np.arange(-0.5, original_map.shape[0], 1), [])
    plt.xlim(-0.5, original_map.shape[1] - 0.5)
    plt.ylim(-0.5, original_map.shape[0] - 0.5)
    plt.show()

def generate_random_non_colliding_point(buffered_map):
    free_cells_row_indices, free_cells_col_indices = np.nonzero(buffered_map == 0)

    free_cells = list(zip(free_cells_row_indices, free_cells_col_indices))

    if not free_cells:
        raise ValueError("No free cells available in the map to generate a point.")

    return random.choice(free_cells)

def resample_path_to_length(path, target_length):
    if not path or len(path) < 2:
        # Handle cases where path is too short or empty
        if not path: # Empty path
            return [[0.0, 0.0]] * target_length # Return a path of zeros for consistency, adjust as needed
        else: # Path with one point
            return [list(path[0])] * target_length # Repeat the single point

    path = np.array(path)
    # Calculate cumulative distance along the path
    distances = np.cumsum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1)))
    distances = np.insert(distances, 0, 0) # Add 0 for the first point

    # If the original path is already shorter or equal to target_length,
    # interpolation will still work, but we need to ensure unique distances for interp1d
    if len(np.unique(distances)) < 2: # All distances are the same (e.g., path is just two identical points)
        return [list(path[0])] * target_length # Repeat the start point or handle as an error

    # Create interpolation functions for x and y coordinates
    # Use fill_value="extrapolate" to handle cases where new_distances might slightly exceed
    # the original max distance due to floating point precision or if target_length is 1.
    interp_x = interp1d(distances, path[:, 0], kind='linear', fill_value="extrapolate")
    interp_y = interp1d(distances, path[:, 1], kind='linear', fill_value="extrapolate")

    # Generate equally spaced distances for the new path
    new_distances = np.linspace(0, distances[-1], target_length)

    # Interpolate to find new x and y coordinates
    resampled_x = interp_x(new_distances)
    resampled_y = interp_y(new_distances)

    return np.column_stack((resampled_x, resampled_y)).tolist()

if __name__ == "__main__":
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

    print(f"Start: {start}, Goal: {goal}")

    astar = AStar(buffered_map, start, goal, step_size=1.0)
    path = astar.generate_path()

    if path:
        print("Path found!")
    
    else:
        print("No path found.")

    plot_obstacle_map(original_map, buffered_map, path)

# Notes
'''
Maybe a change I would like to have (and also see what happens if I do so)
would be find out what is the
'''