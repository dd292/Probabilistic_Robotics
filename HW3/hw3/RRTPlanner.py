import numpy as np
from RRTTree import RRTTree
import time
import sys

class RRTPlanner(object):

    def __init__(self, planning_env, bias = 0.05, eta = 1.0, max_iter = 10000):
        self.env = planning_env         # Map Environment
        self.tree = RRTTree(self.env)
        self.bias = bias                # Goal Bias
        self.max_iter = max_iter        # Max Iterations
        self.eta = eta                  # Distance to extend

    def Plan(self, start_config, goal_config):
        # TODO: YOUR IMPLEMENTATION HERE
        start_config= start_config.ravel()
        goal_config= goal_config.ravel()
        plan_time = time.time()
        plan = []
        # Start with adding the start configuration to the tree.
        self.tree.AddVertex(start_config)
        for i in range(self.max_iter):


            r= self.sample(goal_config)
            vid,vertex = self.tree.GetNearestVertex(r)
            if (r.ravel()==vertex).all():
                continue

            new= self.extend(vertex,r)
            if (not self.env.edge_validity_checker(vertex.reshape((2,1)),new.reshape((2,1)))):
                continue
            #print(news)



            new_cost= self.tree.costs[vid]+self.env.compute_distance(vertex, new)
            nid= self.tree.AddVertex(new,new_cost)
            self.tree.AddEdge(vid,nid)
            if(new==goal_config.ravel()).all():
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

        return plan.T

    def extend(self, x_near, x_rand):
        # TODO: YOUR IMPLEMENTATION HERE
        start = x_near.ravel()
        end = x_rand.ravel()
        v= end-start
        u= v/(np.sqrt(np.sum(v**2)))
        d= self.env.compute_distance(start,end)*self.eta
        point = start+ u*d
        for i in range(2):
            point[i]= round(point[i],2)
        return point


    def sample(self, goal):
        # Sample random point from map
        if np.random.uniform() < self.bias:
            return goal

        return self.env.sample()
