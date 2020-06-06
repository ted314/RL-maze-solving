#进度记录(5-29)：把状态空间的数据结构换成树


import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image
import time

#生成5x6的测试迷宫
maze=np.array([[-1,1,-1,-1,-1],[-1,1,-1,1,-1],[-1,1,-1,1,-1],
	[-1,1,1,1,1],[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1]])

#reward=np.array([-10,0,-10,-10,-10],[-10,0,-10,0,-10],
#	[-10,0,-10,0,-10],[-10,0,0,0,10],[-10,-10,-10,-10,-10])
start=(0,1)
out=(3,4)
gamma=0.8  #折扣因子，控制奖励延时

#注意mat是用mat.shape()
rows,cols=maze.shape

#得到可行状态后，返回状态转移概率 with action
def get_p(x,y):
	up=right=down=left=0
	if maze[x-1][y]>0 and x>0:
		up=1
	if y<(cols-1):
		if maze[x][y+1]>0:
			right=1
	if x<(rows-1):
		if maze[x+1][y]>0:
			down=1
	if maze[x][y-1]>0 and y>0:
		left=1
	p=1/(up+right+down+left)
	ans=[up,right,down,left]
	for i in range(4):
		if ans[i]!=0:
			ans[i]=p
	return ans

#输入迷宫，得到所有可行状态、每一个状态可以采取的动作、初始奖励
#states、actions、reward都用线性表储存
def get_env(maze):
	states=[]
	actions=[]
	reward=np.mat(np.zeros((rows+1,cols+1)))   #多一行一列，防止索引溢出
	for x in range(rows):
		for y in range(cols):
			if maze[x,y]==1:
				states.append((x,y))
				actions.append(get_p(x,y))
				if (x,y)==out:
					reward[x,y]=10
	return states,actions,reward

#输入所有的状态以及其对应的动作，得到每个状态的奖励
#num:迭代次数
def policy_eval(states,actions,reward,num=10):
	for m in range(num):
		j=0
		for (x,y) in states:
			reward[x,y]=reward[x,y]+gamma*(reward[x-1,y]*actions[j][0]+reward[x,y+1]*actions[j][1]+
				reward[x+1,y]*actions[j][2]+reward[x,y-1]*actions[j][3])
			j+=1

	return reward

def policy_update(states,actions,reward):
	j=0
	for (x,y) in states:
		reward_space=[reward[x-1,y],reward[x,y+1],reward[x+1,y],reward[x,y-1]]
		choice=reward_space.index(max(reward_space))        #没有考虑两个方向值相等的情况
		for i in range(4):
			if i==choice:
				actions[j][i]=1
			else:
				actions[j][i]=0
		j+=1
	return actions

#以start为初始状态，读取action_space来行动，把走过的格子的maze的值设为10
#走到终点时，返回已解的maze
#若100步后仍然没有到达，报错并跳出循环
def walk(start,maze,action_space):
	(x,y)=start
	maze[x][y]=5
	step=0
	while step<100:
		if (x,y)==out:
			maze[x][y]=15
			return maze
		maze[x][y]=10
		if action_space[x,y]==1:
			x-=1
		elif action_space[x,y]==2:
			y+=1
		elif action_space[x,y]==3:
			x+=1
		elif action_space[x,y]==4:
			y-=1
		step+=1
	return maze

#判断当前的环境是否已解
#已解：返回action_space（矩阵,1向上，2向右，3向下，4向左）
#未解：返回0
def end_test(start,maze,actions,states):
	action_space=np.mat(np.zeros((rows+1,cols+1)))
	j=0
	for act in actions:
		(x,y)=states[j]
		if max(act)==1:
			pass
			if act.index(max(act))==0:
				action_space[x,y]=1
			elif act.index(max(act))==1:
				action_space[x,y]=2
			elif act.index(max(act))==2:
				action_space[x,y]=3
			elif act.index(max(act))==3:
				action_space[x,y]=4
		else:
			return 0
		j+=1
	return action_space
	

s,a,r=get_env(maze)
for i in range(5):
	r=policy_eval(s,a,r)
	a=policy_update(s,a,r)
act=end_test(start,maze,a,s)
maze=walk(start,maze,act)
plt.matshow(act)
plt.matshow(r)
plt.matshow(maze)