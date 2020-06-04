#mean and SD script

import argparse
import numpy as np
import matplotlib.pyplot as plt

from MapEnvironment import MapEnvironment
from CarEnvironment import CarEnvironment
from AStarPlanner import AStarPlanner
from RRTPlanner import RRTPlanner
from RRTStarPlanner import RRTStarPlanner
from RRTPlannerNonholonomic import RRTPlannerNonholonomic

def main(planning_env, planner, start, goal, argplan):

    # Notify.
    print('starting')

    #planning_env.init_visualizer()

    # Plan
    plan,cost,plan_time,goal_bool = planner.Plan(start, goal)

    # Visualize the final path.
    # tree = None
    # visited = None
    # if argplan != 'astar':
    #     tree = planner.tree
    # else:
    #     visited = planner.visited
    #planning_env.visualize_plan(plan, tree, visited)
    #plt.show()
    return cost,plan_time,goal_bool


if __name__ == "__main__":
    times=[]
    costs=[]
    for i in range(10):

        start= np.array([40 ,100, 4.71]).reshape((3,1))
        goal= np.array([350, 150, 1.57]).reshape((3,1))
        planning_env = CarEnvironment('car_map.txt',start ,goal)
        # Next setup the planner
        bias= 0.05
        eta= 0.5
        plan='nonholrrt'
        if plan == 'rrt':
            planner = RRTPlanner(planning_env, bias, eta)
        elif plan == 'rrtstar':
            planner = RRTStarPlanner(planning_env, bias, eta)
        elif plan == 'nonholrrt':
        	planner = RRTPlannerNonholonomic(planning_env, bias)
        else:
            print('Unknown planner option')
            exit(0)

        cost, plan_time,goal_bool= main(planning_env, planner, start, goal, plan)
        if (not goal_bool):
            i-=1
            continue
        costs.append(cost)
        times.append(plan_time)
    costs= np.asarray(costs)
    times=np.asarray(times)
    print("cost_mean",costs.mean())
    print("cost_SD",costs.std())
    print("time_mean",times.mean())
    print("time_SD",times.std())
