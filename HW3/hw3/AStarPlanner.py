import numpy as np
import heapq
import sys


class AStarPlanner(object):    
    def __init__(self, planning_env, epsilon):
        self.env = planning_env
        self.nodes = {}
        self.epsilon = epsilon
        self.visited = np.zeros(self.env.map.shape)
    
    def Plan(self, start_config, goal_config):
        # TODO: YOUR IMPLEMENTATION HERE
        
        
        start_config= start_config.astype(int).ravel()
        goal_config= goal_config.astype(int).ravel()
        q = PQ()
        plan = []
        parent= {}
        q.push((0,start_config))
        cost= np.ones(self.env.map.shape)*float('inf')
        cost[start_config[0],start_config[1]]=0
        state_count = 0
        parent[self.coord2id(start_config)]=None
        while (not q.empty()):
            
            current_node= q.pop()
            prev_cost= current_node[0]
            current= current_node[1]
            
            self.visited[current[0]][current[1]]=1
            #print('current',current)
            
            if (current==goal_config.ravel()).all():
                print('goal_found')
                #cost_goal = 0
                temp= current
                next_parent= parent[self.coord2id(temp)]
                while(next_parent!=start_config).all():
                    cost_goal= cost[current[0]][current[1]]
                    plan.append(temp)
                    temp= next_parent.copy()
                    next_parent= parent[self.coord2id(temp)]
                plan.append(start_config)
                break
            state_count+=1
            neighbours= self.get_successors(current)
            for item in neighbours:
                ix= item[0]
                iy= item[1]
                g_score= np.asscalar(cost[current[0]][current[1]]+ self.env.compute_distance(current,item))
                if (self.visited[ix][iy]!=1 and cost[ix][iy]>g_score):
                    f_score= g_score+self.epsilon*self.env.compute_distance(item,goal_config)
                    cost[ix][iy]=g_score
                    parent[self.coord2id(item)]=current
                    q.push((f_score,[ix,iy]))

        print("States Expanded: %d" % state_count)
        print("Cost: %f" % cost_goal)
        plan=np.asarray(plan)
        plan= plan[::-1]
        
        return plan.T

    def coord2id(self, coord):
        return coord[0]*self.env.map.shape[1]+coord[1]
    def id2coord(self,ind):
        row= int(ind/self.env.map.shape[1])
        return [row, ind-row*self.env.map.shape[1]]
        
        
            
    def get_successors(self,config):
        successor_pred = [[1,-1],[-1,1],[1,1],[-1,-1],[1,0],[0,1],[-1,0],[0,-1]]
        
        neighbours=[]
        for i in successor_pred:
            succ=np.zeros(2)
            succ[0]=int(config[0]+i[0])
            succ[1]=int(config[1]+i[1])
            if (self.env.state_validity_checker(succ)):
                neighbours.append(succ.astype(int))
        return neighbours
        
class PQ:
    
    def __init__(self):
        self.Q=[]
        
    def push(self,elem):
        self.Q.append(elem)
        
    def pop(self):
        if self.empty():
            return None
        min_cost= float('inf')
        for iter1,i in enumerate(self.Q):
            cost= i[0]
            if cost<min_cost:
                min_cost=cost
                ret_val = i
                index=iter1
                
        del self.Q[index]
        return ret_val
    def empty(self):
        return len(self.Q)==0