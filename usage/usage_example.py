import sys
sys.path.insert(0, '../')

from utils.utils import rrt_mode as mode
from utils.utils import StateSpace, Obstacle
from RRT import RRT
import numpy as np

if __name__=="__main__":
  x0 = np.array([-0.5, -1])
  xd = np.array([0.3, 1.8])

  obs1 = Obstacle("brick", (np.array([0, 0.4]), np.array([0.3, -0.2])))
  obs2 = Obstacle("sphere", (np.array([-0.5, 0.5]), 0.3))
  obs3 = Obstacle("sphere", (np.array([-1, -0.5]), 0.4))
  obs4 = Obstacle("brick", (np.array([-.5, 1.5]), np.array([1, 1.25])))
  
  ss = StateSpace(stateRange=np.array([[-2, -2], [2, 2]]), obs=[obs1, obs2, obs3, obs4])
  path_planner = RRT(ss, mode=mode.rrt_star, K=10)
  path_planner.search_path(x0, xd)
  path_planner.show_path(demo=True, speed=10)
  
