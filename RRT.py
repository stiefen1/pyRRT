import numpy as np
from utils.utils import kNN, Tree, Node, StateSpace, Obstacle
from utils.utils import rrt_mode as mode
import matplotlib.pyplot as plt
from time import sleep, time

class RRT:
  def __init__(self, stateSpace, mode=mode.rrt, K=10, stepSize=0.3, maxStepSize=None, maxIter=1000, tol=0.2):
    if maxStepSize is None:
      maxStepSize=2*stepSize
      
    self._stateSpace, self._mode, self._K = stateSpace, mode, K
    self._stepSize, self._maxStepSize, self._maxIter, self._tol = stepSize, maxStepSize, maxIter, tol
    self._dim = stateSpace.dim
    
  def search_path(self, x0, xd, tol=None):
    self._tree = Tree(x0)
    self._x0 = x0
    self._xd = xd

    if tol is not None:
      self._tol = tol

    start = time()
    for i in range(1, self._maxIter): # Main loop : Create nodes until goal is found
      collision = True
      while collision:                # Secondary loop : Create a node until it is not colliding anything
        x_sampled = self._stateSpace.sample_point(exclude_obstacles=False) 
        # Get the closest node (parent node) from the sampled node
        knn = kNN(X=self._tree.X, K=self._K)
        x_parent, cost_from_parent, idx_parent = knn.to(x_sampled, K=1)
        # Compute the candidate node
        x_new = x_parent + self._stepSize * (x_sampled - x_parent) / cost_from_parent
        # Check if the candidate is colliding the environment
        if not(self._stateSpace.collides(x_new) or self._stateSpace.collides(x_parent, x_new)):
          collision = False

      node_parent, _ = self._tree.get_node(idx_parent)
      cost_tot_to_x_new = self._stepSize + node_parent.cost

      # RRT* check if using one of the knn of x_candidate decreases the cost
      if self._mode == mode.rrt_star and i > 1:
        # Get the K Nearest Neighboors to the new point
        x_knn, _, idx_knn = knn.to(x_new)
        # Compute the total cost to reach x_new for the k nearest neighboors
        cost_tot_to_x_new_list = self._tree.get_cost_tot_to(x_new, idx_knn)
        
        collision=True
        while np.min(cost_tot_to_x_new_list) != float('inf') and collision:
          # New parent index is the one with smallest cost
          j_min = np.argmin(cost_tot_to_x_new_list)
          cost_tot_to_x_new = cost_tot_to_x_new_list[j_min]
          #print(cost_tot_to_x_candidate, j_min)
          idx_parent = idx_knn[j_min]
          # Set current best value to infinity to avoid selecting it again
          cost_tot_to_x_new_list[j_min] = float('inf')
          # Check for obstacles
          collision = self._stateSpace.collides(x_knn[j_min], x_new)

      if not(collision):
        new_node = Node(x_new, idx_parent, cost_tot_to_x_new)
        self._tree.add_node(new_node)

      goal_reached, x_intersection = self._goal_reached(new_node)
      if goal_reached:
        dist_to_goal = np.linalg.norm(x_new - xd)
        cost_parent = self._tree.nodes[idx_parent].cost
        new_node = Node(x_intersection, idx_parent, self._tol + cost_parent)
        self._tree.remove_node(-1)
        self._tree.add_node(new_node)
        break

      
    self._iter = i+1

    # Add desired position as final node
    if i+1 < self._maxIter:
      error = np.linalg.norm(x_new-xd)
      final_node = Node(xd, i, error + new_node.cost)
      self._tree.add_node(final_node)
      time_to_converge = time()-start
      print("Path found in {:.2f} seconds !".format(time_to_converge))

    else:
      print("Unable to find a valid path !")
      time_to_converge = float('inf')

    return self._tree.path, time_to_converge

  def _goal_reached(self, new_node):
    node_parent, _ = self._tree.get_node(new_node.parent)
    x1 = node_parent.x
    x2 = new_node.x
    dist_x1_x2 = np.linalg.norm(x2 - x1)
    u = (x2 - x1) / dist_x1_x2
    dx1xc = x1 - self._xd
    a = (1/self._tol**2) * u @ u.T
    b = (2/self._tol**2) * (dx1xc @ u.T)
    c = (1/self._tol**2) * (dx1xc @ dx1xc.T) - 1
    Delta = b**2 - 4*a*c

    if Delta > 0:
      lambda1 = (-b - np.sqrt(Delta)) / (2*a)
      lambda2 = (-b + np.sqrt(Delta)) / (2*a)
      if 0 <= lambda1 <= dist_x1_x2:
        return True, lambda1 * u + x1
      if 0 <= lambda2 <= dist_x1_x2:
        return True, lambda2 * u + x1
    elif Delta == 0:
      lambda1 = -b/(2*a)
      if 0 <= lambda1 <= dist_x1_x2:
        return True, lambda1 * u + x1

    return False, None

  def show_path(self, demo=False, speed=1):
    dim = self._tree.dim
    N = len(self._tree.path)
    X = np.zeros((dim, N))
    Ncolors = 50
    colors = plt.cm.rainbow(np.linspace(0, 1, Ncolors))
    cost_max = self._tree._nodes[-1].cost / (self._stepSize**(2/5))
    best_cost = np.sqrt((self._xd - self._x0) @ (self._xd - self._x0).T)
    
    for n, node in enumerate(self._tree.path):
        X[:, n] = node.x

    if demo:
      plt.figure()
      ax = plt.gca()
      plt.grid()
      for obs in self._stateSpace.obs:
        ax.add_patch(obs.patch_to_plot)

      plt.plot(self._xd[0], self._xd[1], 'x', color='red')
      plt.plot(self._x0[0], self._x0[1], 'x', color='red')
      ax.add_patch(plt.Circle((self._xd[0], self._xd[1]), self._tol, linestyle='--', facecolor='white', edgecolor='red'))
        
      for i, node in enumerate(self._tree._nodes[1::]):
        start = time()
        plt.plot([node.x[0], self._tree._nodes[node.parent].x[0]],
                 [node.x[1], self._tree._nodes[node.parent].x[1]],
                 color=colors[int(min(node.cost, cost_max) * (Ncolors-1) / cost_max)])
        plt.xlim(self._stateSpace.range[0, 0], self._stateSpace.range[1, 0])
        plt.ylim(self._stateSpace.range[0, 1], self._stateSpace.range[1, 1])
        plt.title("Three after {}/{} iterations".format(i, self._maxIter))
        dt = time() - start
        plt.pause(max(0.1/speed-dt, 0.001))
      plt.draw()

    else:
      plt.figure()
      plt.grid()

    
    plt.plot(X[0, :], X[1, :], 'black')
    plt.xlim(self._stateSpace.range[0, 0], self._stateSpace.range[1, 0])
    plt.ylim(self._stateSpace.range[0, 1], self._stateSpace.range[1, 1])
    plt.title("Path after {}/{} iterations - Cost : {:.2f}/{:.2f}".format(self._iter, self._maxIter, self._tree.path[0].cost, best_cost))
    
    plt.show()
          
  @property
  def stateSpace(self):
    return self._stateSpace

  @property
  def K(self):
    return self._K

  @property
  def mode(self):
    return self._mode

  @property
  def path(self):
    return self._tree.path

################################## USAGE EXAMPLE ###################################
  
if __name__=="__main__":
  x0 = np.array([-0.5, -1])
  xd = np.array([0.3, 1.8])
  obs1 = Obstacle("brick", (np.array([0, 0.4]), np.array([0.3, -0.2])))
  obs2 = Obstacle("sphere", (np.array([-0.5, 0.5]), 0.3))
  obs3 = Obstacle("sphere", (np.array([-1, -0.5]), 0.4))
  obs4 = Obstacle("brick", (np.array([-.5, 1.5]), np.array([1, 1.25])))

  ss = StateSpace(stateRange=np.array([[-2, -2], [2, 2]]), obs=[obs1, obs2, obs3, obs4])
  path_planner = RRT(ss, mode=mode.rrt_star, K=10, tol=0.15)
  path_planner.search_path(x0, xd)
  path_planner.show_path(demo=True, speed=10)
  
  
