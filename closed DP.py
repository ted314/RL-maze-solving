from random import randint, choice
from enum import Enum
import numpy as np 
import matplotlib.pyplot as plt 

#生成11x15的迷宫
maze=np.mat([[-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
	[-1,1,1,1,1,1,1,1,1,1,-1],
	[-1,1,-1,-1,-1,1,-1,-1,1,-1,-1],
	[-1,1,-1,-1,-1,-1,-1,-1,1,1,-1],
	[-1,1,1,-1,1,1,1,1,1,-1,-1],
	[-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1],
	[-1,1,-1,-1,1,-1,1,-1,1,1,-1],
	[-1,1,-1,-1,1,-1,1,-1,1,1,-1],
	[-1,1,1,1,1,1,1,-1,1,1,-1],
	[-1,1,-1,-1,-1,1,-1,-1,-1,-1,-1],
	[-1,1,-1,-1,-1,1,1,1,1,1,-1],
	[-1,1,-1,-1,-1,1,-1,-1,1,1,-1],
	[-1,1,-1,-1,1,1,-1,-1,1,-1,-1],
	[-1,1,1,1,-1,1,1,-1,1,1,1],
	[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
	])

start=(0,1)
out=(13,10)
gamma=0.2  #折扣因子，控制奖励延时
discount=0.1  #缩小因子，控制奖励值<1000

#注意mat是用mat.shape()
rows,cols=maze.shape

#得到可行状态后，返回状态转移概率 with action
def get_p(x,y):
	up=right=down=left=0
	if maze[x-1,y]>0 and x>0:
		up=1
	if y<(cols-1):
		if maze[x,y+1]>0:
			right=1
	if x<(rows-1):
		if maze[x+1,y]>0:
			down=1
	if maze[x,y-1]>0 and y>0:
		left=1
	p=1/(up+right+down+left)
	ans=[up,right,down,left]
	for i in range(4):
		if ans[i]!=0:
			ans[i]=p
	return ans

#输入迷宫，得到所有可行状态、每一个状态可以采取的动作、初始奖励
#states、actions都用线性表储存，reward用矩阵
def get_env(maze,out):
	states=[]
	actions=[]
	reward=np.mat(np.zeros((rows+1,cols+1)))   #多一行一列，防止索引溢出
	for x in range(rows):
		for y in range(cols):
			if maze[x,y]==1:
				states.append((x,y))
				actions.append(get_p(x,y))
				if (x,y)==out:
					reward[x,y]=1
	start=choice(states)
	while start==out:
		start=choice(states)
	return start,states,actions,reward

#输入所有的状态以及其对应的动作，得到每个状态的奖励
#num:迭代次数
def policy_eval(states,actions,reward,num=25):
	for m in range(num):
		j=0
		for (x,y) in states:
			reward[x,y]=reward[x,y]+gamma*(reward[x-1,y]*actions[j][0]+reward[x,y+1]*actions[j][1]+
				reward[x+1,y]*actions[j][2]+reward[x,y-1]*actions[j][3])
			j+=1
		if reward.max()>1000:
			reward=reward*discount  #减小奖励函数的绝对值，有助于惩罚
		
	return reward

def policy_update(states,actions,reward):
	j=0
	for (x,y) in states:
		reward_space=[reward[x-1,y],reward[x,y+1],reward[x+1,y],reward[x,y-1]]
		"""
		add=sum(reward_space)
		for i in range(4):
			if reward_space[i]!=0:
				actions[j][i]=reward_space[i]/add
		j+=1
		"""
		choice=reward_space.index(max(reward_space))     #贪心选择奖励最大的，没有考虑两个方向值相等的情况
		for i in range(4):
			if i==choice:
				actions[j][i]=1
			else:
				actions[j][i]=0
		j+=1
		
	return actions

def random_update(states,actions,reward):
	j=0
	for (x,y) in states:
		reward_space=[reward[x-1,y],reward[x,y+1],reward[x+1,y],reward[x,y-1]]
		choi_space=[]
		for re in reward_space:
			if re!=0:
				choi_space.append(re)
		choi=reward_space.index(choice(choi_space))     #随机选择一个为自己的action
		for i in range(4):
			if i==choi:
				actions[j][i]=1
			else:
				actions[j][i]=0
		j+=1
	return actions

#先生成actions_space，再根据它来行动
#路径存储在一个列表中，每次append的时候检查是否重复（在重复点施加惩罚）
#若走出迷宫，返回路径
#若出现重复，返回惩罚后的reward
def walk(start,maze,states,actions,reward):
	action_space=np.mat(np.zeros((rows+1,cols+1)))
	j=0
	for act in actions:
		(x,y)=states[j]
		if act.index(max(act))==0:
			action_space[x,y]=1
		elif act.index(max(act))==1:
			action_space[x,y]=2
		elif act.index(max(act))==2:
			action_space[x,y]=3
		elif act.index(max(act))==3:
			action_space[x,y]=4
		j+=1

	(x,y)=start
	step=0
	solution=[]
	tag=0   #0表示没走出去
	while step<100:
		if (x,y)==out:
			tag=1
			return reward,action_space,solution,tag  #1表示已解
		if (x,y) in solution:
			reward[x,y]-=100
			return reward,action_space,solution,tag
		else:
			solution.append((x,y))
		if action_space[x,y]==1:
			x-=1
		elif action_space[x,y]==2:
			y+=1
		elif action_space[x,y]==3:
			x+=1
		elif action_space[x,y]==4:
			y-=1
		step+=1
	tag=2  #2表示步数超出限制
	return reward,action_space,solution,tag

#显示最初的maze、当前的reward、solution(maze)
#reward需要处理一下以便观察
def display(maze,states,solution,reward,actions):
	plt.matshow(maze,cmap='CMRmap')
	plt.title('Orignal Maze')
	for (x,y) in solution:
		maze[x,y]=0.5
	plt.matshow(maze,cmap='CMRmap')
	plt.title('Solved Maze')
	for i in range(5):
		reward=policy_eval(states,actions,reward)
		actions=policy_update(states,actions,reward)
	plt.matshow(reward,cmap='Blues')
	plt.title('Reward')

start,s,a,r=get_env(maze,out)
for i in range(60):
	r=policy_eval(s,a,r)
	if i%2==0:
		a=random_update(s,a,r)
	else:
		a=policy_update(s,a,r)
r,act,solu,tag=walk(start,maze,s,a,r)
plt.matshow(act,cmap='CMRmap')
plt.title('Action-Space')
plt.matshow(r,cmap='Blues')
plt.title('Reward')

"""
counter=0
tag=0
while tag==0 and counter<50:
	r=policy_eval(s,a,r)
	a=policy_update(s,a,r)
	r,act,solu,tag=walk(start,maze,s,a,r)
	counter+=1
plt.matshow(act,cmap='CMRmap')
plt.title('Action-Space')
print("迷宫已解，共尝试： %s 次" %counter)
display(maze,s,solu,r,a)
"""