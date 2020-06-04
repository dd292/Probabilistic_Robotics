import numpy as np
from RRTTree import RRTTree
import time
import copy

class RRTStarPlanner(object):

    def __init__(self, planning_env, bias = 0.05, eta = 1.0, max_iter = 10000):
        self.env = planning_env         # Map Environment
        self.tree = RRTTree(self.env)
        self.bias = bias                # Goal Bias
        self.max_iter = max_iter        # Max Iterations
        self.eta = eta                  # Distance to extend
        self.childs={}
        

    def Plan(self, start_config, goal_config, rad=50):
        # TODO: YOUR IMPLEMENTATION HERE
        start_config= start_config.ravel()
        goal_config= goal_config.ravel()
        plan_time = time.time()
        plan = []
        # Start with adding the start configuration to the tree.
        self.tree.AddVertex(start_config)
        goal=False
        for i in range(self.max_iter):
            if (i%1000==0):
                print(i)
            
            r= self.sample(goal_config)
            vid,vertex = self.tree.GetNearestVertex(r)
            if (r.ravel()==vertex).all():
                continue
            
            new= self.extend(vertex,r)
            
            # RRtstar algo
            if (self.env.edge_validity_checker(vertex.reshape((2,1)),new.reshape((2,1)))):
                # if obstacle free
                X_nearID,X_near= self.tree.GetNNInRad(new,rad)
                x_min= vertex
                c_min=self.tree.costs[vid]+self.env.compute_distance(vertex, new) 
                min_ID= vid
                for close_neigh,cID in zip(X_near,X_nearID):
                    if (self.env.edge_validity_checker(close_neigh.reshape((2,1)),new.reshape((2,1)))):
                        new_cost= self.tree.costs[cID]+self.env.compute_distance(close_neigh, new)
                        
                        if (new_cost<c_min):
                            c_min= new_cost
                            x_min= close_neigh
                            min_ID= cID
                nid= self.tree.AddVertex(new,c_min)    
                self.tree.AddEdge(min_ID,nid)
                self.add_parents(min_ID,nid)
                #rewiring the tree
                for close_neigh,cID in zip(X_near,X_nearID):
                    if (self.env.edge_validity_checker(close_neigh.reshape((2,1)),new.reshape((2,1)))):
                        new_cost= c_min+self.env.compute_distance(close_neigh, new)
                        if (new_cost<self.tree.costs[cID]):
                            # update_edge
                            prev_parent= self.tree.edges[cID]
                            self.childs[prev_parent].remove(cID)
                            self.tree.AddEdge(nid,cID)
                            self.add_parents(nid,cID)
                            self.update_cost_tree(cID,self.tree.costs[cID]-new_cost)                                      
                            
            else:
               continue
            
            if(new==goal_config.ravel()).all():
                print('goal_reached')
                goal= True
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
    def update_cost_tree(self, ID,cost_diff):
        
        queue=[]
        queue.append(ID)
        while(len(queue)!=0):
            new_ID= queue.pop()
            self.tree.costs[new_ID]-=cost_diff
            if (new_ID in self.childs.keys()):
                for i in self.childs[new_ID]:
                    queue.append(i)
            
    def add_parents(self,parent,child):
        if (parent not in self.childs.keys()):
            self.childs[parent]=[]
        self.childs[parent].append(child)
        
    # def calc_cost(self,ID):#to calculate cost from start node every time because costs are chainging when rewiring the tree
    #     total_cost=0
    #     prev_ID= ID
    #     while(prev_ID!=0):
    #         next_ID= self.tree.edges[prev_ID]
    #         total_cost+=self.env.compute_distance(self.tree.vertices[prev_ID],self.tree.vertices[next_ID] )
    #         prev_ID= next_ID
    #     return total_cost
            
    def sample(self, goal):
        # Sample random point from map
        if np.random.uniform() < self.bias:
            return goal

        return self.env.sample()
