import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Define the matrix representing the gridworld, where 1s are walls and 0s are free spaces
# If you look more closely, this 9x9 grid is specifically designed to be cutted at any size (2x2, 3x3, ..., 9x9)
# while still maintaining a valid path from the top-left corner (start) to the bottom-right corner (goal).
# NOTE: All the values on the principal diagonal are 0, thus ensuring that the goal does not overlap with a wall. 

GRID_WORD = np.array([
    [0, 1, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 0, 1],
    [0, 0, 1, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 1, 0, 1],
    [0, 0, 0, 1, 0, 0, 0, 0, 0]
])

class POMDPGridworld(gym.Env):
    '''
    A partially observable gridworld environment where the agent receives binary observations about adjacent walls 
    instead of its exact coordinates. The agent must navigate from the top-left corner to the bottom-right corner 
    while avoiding walls. The environment is designed to be scalable, allowing for different grid sizes 
    (from 2x2 to 9x9) while maintaining a valid path to the goal.

    Input:
    - size: The size of the grid (size x size). Must be between 2 and 9.
    - max_steps: The maximum number of steps before the episode is truncated.

    Functions:
    - reset(): Resets the environment to the initial state and returns the initial observation.
    - step(action): Takes an action and returns the new observation, reward, termination status, and truncation status.
    - visualize_grid(): Prints a visual representation of the grid, showing the agent's position, the goal, and 
                        the walls.
    '''
    def __init__(self, size=5, max_steps=500):
        super(POMDPGridworld, self).__init__()

        try:
            size = int(size)
            if size < 2 or size > 9:
                raise ValueError(f"Invalid size: {size}. Size must be between 2 and 9.")
        except ValueError as e:
            print(f"Invalid size: {size}. {e}")
            raise

        self.size = size
        self.max_steps = max_steps
        self.current_step = 0
        
        # ACTION: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        self.action_space = spaces.Discrete(4)
        
        # POMDP OBSERVATION: 4 binary sensors for adjacent walls (0=free, 1=wall)
        # Instead of returning coordinates (x,y), return what's around the agent:
        # [wall_up, wall_right, wall_down, wall_left]
        self.observation_space = spaces.MultiBinary(4)
        
        # INTERNAL STATE (hidden from the agent)
        self.agent_pos = np.zeros(2, dtype=int)  # Agent's position (x,y)
        self.goal_pos = np.array([size-1, size-1], dtype=int)  # Goal position
        self.grid = GRID_WORD[:self.size, :self.size]  # Walls based on the GRID_WORD matrix

    def reset(self, **kwargs):
        super().reset(**kwargs)
        # Reset the agent to the starting position (0,0) and reset the step counter
        self.agent_pos = np.zeros(2, dtype=int)
        self.current_step = 0
        
        # Return the OBSERVATION (not the internal state)
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        self.current_step += 1
        
        # 1. Update the TRUE state (the coordinates (x,y))
        self._move_agent(action)
        
        # 2. Check if the goal has been reached and decide the reward
        terminated = np.array_equal(self.agent_pos, self.goal_pos)
        reward = 1.0 if terminated else -0.01 # Small time penalty
        
        truncated = self.current_step >= self.max_steps
        
        # 3. Calculate the PARTIAL observation
        obs = self._get_observation()
        
        # Return the observation
        return obs, reward, terminated, truncated, {}

    def _get_observation(self):

        # Initialize the observation with zeros (no walls)
        obs = np.zeros(4, dtype=np.int8)

        # Check for walls in the four adjacent directions and update the observation accordingly
        if self.agent_pos[1] == 0 or self.grid[self.agent_pos[1]-1, self.agent_pos[0]] == 1:
            obs[0] = 1
        if self.agent_pos[0] == 0 or self.grid[self.agent_pos[1], self.agent_pos[0]-1] == 1:
            obs[3] = 1
        if self.agent_pos[1] == self.size-1 or self.grid[self.agent_pos[1]+1, self.agent_pos[0]] == 1:
            obs[2] = 1
        if self.agent_pos[0] == self.size-1 or self.grid[self.agent_pos[1], self.agent_pos[0]+1] == 1:
            obs[1] = 1
        
        # Return the observation
        return obs
    
    def _move_agent(self, action):

        # Be sure to check for a valid action
        try:
            action = int(action)
            if action == 0 and self.agent_pos[1] > 0:  # Up
                if self.grid[self.agent_pos[1]-1, self.agent_pos[0]] == 0:
                    self.agent_pos[1] -= 1
            elif action == 1 and self.agent_pos[0] < self.size - 1:  # Right
                if self.grid[self.agent_pos[1], self.agent_pos[0]+1] == 0:
                    self.agent_pos[0] += 1
            elif action == 2 and self.agent_pos[1] < self.size - 1:  # Down
                if self.grid[self.agent_pos[1]+1, self.agent_pos[0]] == 0:
                    self.agent_pos[1] += 1
            elif action == 3 and self.agent_pos[0] > 0:  # Left
                if self.grid[self.agent_pos[1], self.agent_pos[0]-1] == 0:
                    self.agent_pos[0] -= 1
        except ValueError:
            print(f"Invalid action: {action}. Action must be an integer between 0 and {self.action_space.n - 1}.") # type: ignore

    def visualize_grid(self):
        grid = np.full((self.size, self.size), '-')
        grid[self.goal_pos[1], self.goal_pos[0]] = 'G'  # Goal
        for y in range(self.size):
            for x in range(self.size):
                if self.grid[y, x] == 1:
                    grid[y, x] = 'W'  # Wall
        grid[self.agent_pos[1], self.agent_pos[0]] = 'A'  # Agent
        print("\n".join("".join(row) for row in grid))

if __name__ == "__main__":
    '''
    Example usage of the POMDPGridworld environment. 
    This code initializes the environment, resets it to get the initial observation, and then takes 
    a couple of actions while printing the resulting observations, rewards, and termination status.
    '''
    grid = POMDPGridworld(size=2)
    obs, _ = grid.reset()
    print("Initial observation:", obs)
    obs, reward, terminated, _, _ = grid.step(2)  # Try to move down
    print("Observation after action 1 (down):", obs)
    print("Reward:", reward, "Terminated:", terminated)
    obs, reward, terminated, _, _ = grid.step(1)  # Try to move right
    print("Observation after action 2 (right):", obs)
    print("Reward:", reward, "Terminated:", terminated)