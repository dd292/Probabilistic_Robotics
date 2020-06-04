import numpy as np
from RRTTree import RRTTree
import time

class RRTPlannerNonholonomic(object):

    def __init__(self, planning_env, bias=0.05, max_iter=10000, num_control_samples=25):
        self.env = planning_env                 # Car Environment
        self.tree = RRTTree(self.env)
        self.bias = bias                        # Goal Bias
        self.max_iter = max_iter                # Max Iterations
        self.num_control_samples = 25           # Number of controls to sample

    def Plan(self, start_config, goal_config):
        # TODO: YOUR IMPLEMENTATION HERE

        start_config= start_config.ravel()
        goal_config= goal_config.ravel()
        plan_time = time.time()
        plan = []
        # Start with adding the start configuration to the tree.
        start_config= start_config.reshape((3,1))
        goal_config= goal_config.reshape((3,1))
        self.tree.AddVertex(start_config)
        for i in range(self.max_iter):
            if (i%1000==0):
                print(i)
            r= self.sample(goal_config).ravel()
            r= r.reshape((3,1))
            vid,vertex = self.tree.GetNearestVertex(r.reshape((3,1)))
            #vertex= vertex.reshape((3,1))
            if (r.ravel()==vertex).all():
                continue

            new,exec_time= self.extend(vertex,r)
            if (exec_time is None):
                continue
            # if (not self.env.edge_validity_checker(vertex.reshape((3,1)),new.reshape((3,1)))):
            #     continue
            #print('exec_time',exec_time)
            new_cost= self.tree.costs[vid]+ exec_time
            nid= self.tree.AddVertex(new,new_cost)
            self.tree.AddEdge(vid,nid)
            #print('new', new,'goal', goal_config)
            if(self.env.goal_criterion(new, goal_config)):
                print('goal_reached')
                idx=nid
                while (idx!=0):
                    plan.append(self.tree.vertices[idx])
                    idx= self.tree.edges[idx]
                plan.append(self.tree.vertices[idx])
                break
        plan=np.asarray(plan)
        plan= plan[::-1]

        cost = new_cost
        plan_time = time.time() - plan_time

        print("Cost: %f" % cost)
        print("Planning Time: %ds" % plan_time)
        plan= plan.T
        plan= plan.reshape((3,-1))
        return plan

    def extend(self, x_near, x_rand):
        """ Extend method for non-holonomic RRT

            Generate n control samples, with n = self.num_control_samples
            Simulate trajectories with these control samples
            Compute the closest closest trajectory and return the resulting state (and cost)
        """
        # TODO: YOUR IMPLEMENTATION HERE
        min_dist= float('inf')
        min_cost=None
        for i in range(self.num_control_samples):
            vel,angle= self.env.sample_action()
            point,dt= self.env.simulate_car(x_near,x_rand,vel,angle)

            if point is not None and self.env.compute_distance(point,x_rand)<min_dist:
                x_near= point
                min_cost=dt
        x_near= x_near.reshape((3,1))
        for i in range(3):
            x_near[i]= round(x_near[i][0],2)
        return x_near ,min_cost

    def sample(self, goal):
        # Sample random point from map
        if np.random.uniform() < self.bias:
            return goal

        return self.env.sample()
    def compute_distance(self, start_config, end_config):
        th1 = start_config[2,:]; th2 = end_config[2,:]
        ang_diff = np.abs(th1-th2)*180/np.pi
        dsit = np.sqrt((start_config[0,:]-end_config[0,:])**2+ (start_config[1,:]-end_config[1,:])**2+ang_diff**2)
        return dsit
