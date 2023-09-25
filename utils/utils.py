from enum import Enum
import numpy as np
from numpy.random import rand
from matplotlib.patches import Rectangle, Circle

class rrt_mode(Enum):
  rrt = 1
  rrt_star = 2

class obstacle_shape(Enum):
  rectangle = 1
  circle = 2
  triangle = 3

######################## Node Class ########################

class Node:
  def __init__(self, x, parent, cost):
    self._x = x
    self._parent = parent
    self._cost = cost

  def set_parent(self, parent):
    self._parent = parent

  def set_cost(self, cost):
    self._cost = cost

  @property
  def x(self):
    return self._x

  @property
  def parent(self):
    return self._parent

  @property
  def cost(self):
    return self._cost

######################### Tree Class ########################

class Tree:
  def __init__(self, x0):
    self._nodes = []
    self._X = []

    node0 = Node(x0, -1, 0)
    self.add_node(node0)

  def get_node(self, idx):
    if isinstance(idx, (list, np.ndarray)):
      idx = idx[0]
      
    if idx in range(0, len(self._nodes)):
      node = self._nodes[idx]
      return node, (node.x, node.parent, node.cost)
    else:
      ValueError("idx is out of range [" + str(0) + " , " + str(len(self._nodes)) + "]")

  def remove_node(self, idx):
    if isinstance(idx, (list, np.ndarray)):
      idx = idx[0]
      
    if idx in range(0, len(self._nodes)):
      self._nodes.pop(idx)
      self._X.pop(idx)
    elif idx == -1:
      self._nodes.pop(-1)
      self._X.pop(-1)
    else:
      ValueError("idx is out of range [" + str(0) + " , " + str(len(self._nodes)) + "]")
  

  def add_node(self, node, idx=-1):
    if idx == -1:
      idx = len(self._nodes)

    self._nodes.insert(idx, node)
    self._X.insert(idx, node.x)

  def get_cost_tot_to(self, x, idx):
    cost_tot_to_x = np.zeros((len(idx),))
    for k, i in enumerate(idx):
      if i in range(0, len(self._nodes)):
        xi = self._nodes[i].x
        ei = x-xi
        cost_tot_to_x[k] = np.sqrt(ei @ ei.T) + self._nodes[i].cost
      else:
        ValueError("idx " + str(idx[i]) + " is out of range [" + str(0) + " , " + str(len(self._nodes)) + "]")

    return cost_tot_to_x

  @property
  def path(self):
    node = self._nodes[-1]
    path = [node]
    while node.parent != -1:
      node = self._nodes[node.parent]
      path.append(node)
      
    return path

  @property
  def X(self):
    return np.array(self._X)

  @property
  def dim(self):
    return self._nodes[-1].x.shape[0]

  @property
  def nodes(self):
    return self._nodes


######################### k-Nearest Neighboors Algorithm #######################
  
class kNN:
  def __init__(self, X=None, K=None):
    self._X, self._K = X, K

  def to(self, x, K=None):
    if K is None:
      K = self._K
    error = self._X - x                             # Compute error matrix
    dist_2 = np.sqrt(np.diag(error @ error.T))      # Compute squared distance (used as cost)
    K = min([K, dist_2.shape[0]])
    idx = np.argpartition(dist_2, K-1)[:K]  # Get index from K smallest distances

    if K == 1:
      idx = idx[0] 

    return np.array(self._X[idx]), dist_2[idx], idx

  def set_datapoints(self, X):
    self._X = np.array(X)

######################### State Space ######################

class StateSpace:
  def __init__(self, stateRange=None, obs=[], path=None):   
    if stateRange is not None:
      self._dim = stateRange.shape[1]
      self._range = stateRange # numpy array([x1_min, x2_min, ...], [x1_max, x2_max, ...])
      self._obs = obs

    elif path is not None:
      self._load_state_space(path)

  def add_obstacles(self, obs):
    self._obs.append(obs)

  def collides(self, *args):
    if len(args) == 1:
      return self._single_collides(args[0])
    elif len(args) > 1:
      return self._pair_collides(args[0], args[1])
    else:
      ValueError("Invalid argument")

  def _single_collides(self, x):
    for obs in self._obs:
      if obs.single_collides(x):
        return True
    else:
      return False

  def _pair_collides(self, x1, x2):
    for obs in self._obs:
      if obs.pair_collides(x1, x2):
        return True
    else:
      return False

  def sample_point(self, exclude_obstacles=True):
    # Initial guess
    x_sampled = np.array((rand(self._dim)-0.5) * self.range_scaler + self.range_mean)
    if exclude_obstacles:
      while self._single_collides(x_sampled):
        x_sampled = np.array((rand(self._dim)-0.5) * self.range_scaler + self.range_mean)

    return x_sampled

  def load_state_space(self, path):
    pass

  @property
  def obs(self):
    return self._obs

  @property
  def range_mean(self):
    return (self._range[0] + self._range[1])/2.

  @property
  def range_scaler(self):
    return self._range[1] - self._range[0]

  @property
  def range(self):
    return self._range

  @property
  def dim(self):
    return self._dim

######################### Obstacle ######################
  
class Obstacle:
  def __new__(self, shape, pos):
    if shape=="brick":
      return BrickObstacle(pos)
    elif shape=="sphere":
      return SphereObstacle(pos)
    else:
      ValueError("Invalid obstacle type !")

class BrickObstacle:
  def __init__(self, pos):
    self._x_upper_left = pos[0]
    self._x_lower_right = pos[1]
    self._height = pos[0][1] - pos[1][1]
    self._width = pos[1][0] - pos[0][0]
    self._x_lower_left = self._x_upper_left - np.array([0, self._height])
    self._x_upper_right = self._x_upper_left + np.array([self._width, 0])

  def single_collides(self, x):
    for (xi_lower_right, xi_upper_left, xi) in zip(self._x_lower_right, self._x_upper_left, x):
      sign = np.sign(xi_lower_right - xi_upper_left)
      if sign*xi < sign*xi_upper_left or sign*xi > sign*xi_lower_right:
        return False
    return True

  def pair_collides(self, x1, x2):
    u = (x2 - x1)/np.linalg.norm(x2 - x1)
    ey = np.array([0, -1])
    ex = np.array([1, 0])
    V = np.stack((u, -ey), axis=1)
    W = np.stack((u, ex), axis=1)
  
    M = np.hstack([V, np.zeros((u.shape[0], 6))])
    M = np.vstack([M, np.hstack([np.zeros((u.shape[0], 2)), V, np.zeros((u.shape[0], 4))])])
    M = np.vstack([M, np.hstack([np.zeros((u.shape[0], 4)), W, np.zeros((u.shape[0], 2))])])
    M = np.vstack([M, np.hstack([np.zeros((u.shape[0], 6)), W])])

    #print(M)
    x1 = x1[:, np.newaxis] # To get a column vector
    X1 = np.block([
      [x1],
      [x1],
      [x1],
      [x1]])
    
    Xbar = np.block([[self._x_upper_left[:, np.newaxis]],
                     [self._x_upper_right[:, np.newaxis]],
                     [self._x_upper_right[:, np.newaxis]],
                     [self._x_lower_right[:, np.newaxis]]])

    if np.linalg.det(M) > 1e-9:
      out = np.linalg.inv(M) @ (Xbar - X1)
    else: # Means vectors is parallel to obstacle
      return False
    if (0 <= out[1] <= self._height) or (0 <= out[3] <= self._height):
      return True

    if (0 <= out[5] <= self._width) or (0 <= out[7] <= self._width):
      return True

    return False
    

  @property
  def patch_to_plot(self):
    return Rectangle(self._x_lower_left, self._width, self._height, facecolor='grey')


class SphereObstacle:
  def __init__(self, pos):
    self._center = pos[0]
    self._radius = pos[1]

  def single_collides(self, x):
    return np.linalg.norm(self._center - x) <= self._radius

  def pair_collides(self, x1, x2):
    dist_x1_x2 = np.linalg.norm(x2 - x1)
    u = (x2 - x1) / dist_x1_x2
    dx1xc = x1 - self._center
    a = (1/self._radius**2) * u @ u.T
    b = (2/self._radius**2) * (dx1xc @ u.T)
    c = (1/self._radius**2) * (dx1xc @ dx1xc.T) - 1
    Delta = b**2 - 4*a*c

    if Delta > 0:
      lambda1 = (-b + np.sqrt(Delta)) / (2*a)
      lambda2 = (-b - np.sqrt(Delta)) / (2*a)
      if 0 <= lambda1 <= dist_x1_x2 or 0 <= lambda2 <= dist_x1_x2:
        return True
    elif Delta == 0:
      lambda1 = -b/(2*a)
      if 0 <= lambda1 <= dist_x1_x2:
        return True

    return False
    

  @property
  def patch_to_plot(self):
    return Circle(self._center, radius=self._radius, facecolor='grey')

  



  

