from utils import rrt_mode as mode
from utils import stateSpace
from RRT import RRT
import numpy as np

if __name__=="__main__":
  x0 = np.array([-0.5, -1])
  xd = np.array([0.3, 1.8])
  ss = stateSpace(stateRange=np.array([[-1, -2], [1, 2]])) #map_path="maps/roommate_room.npy")
  path_planner = RRT(ss, mode=mode.rrt, K=10)
  path_planner.search_path(x0, xd)
  print(len(path_planner.path))
  path_planner.show_path(show_tree=True, demo=True, speed=1)
  
