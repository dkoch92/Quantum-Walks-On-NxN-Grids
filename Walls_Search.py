import NxN_functions as nn
import Maze_Generator as mg
import numpy as np
import math as m
import scipy as sci
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ----------- # ----------- #----------- #----------- #----------- #----------- #----------- #
'''
This python file is designed to showcase several of the results discussed in the paper 
concerning the searches on geometries with walls.  The modules are as follows:
    
Module 1: Generating a Perfect Square Maze
Module 2: Generating a Grid with Walls
Module 3: Stable Hybrid Search on a Grid with Walls
Module 4: Optimal Hybrid Search on a Grid with Walls
Module 5: Confirming the Optimal Hybrid Speed Probabilistically

Each module is a stand-alone block of code, with a marked area for user input
To run a module, remove the triple ' quotes at the beginning and end
Each module produces a result, either printed to console or a plot
'''
# ----------- #----------- #----------- #----------- #----------- #----------- #----------- #

# ------------------------
#      Functions
# ------------------------

def Visualize_Maze(maze):
    '''
    Input: maze (N-N matrix)
    Creates a visualizarion of the walls present in the matrix maze
    Prints to the console
    '''
    shapes = ['','_|','_|','_ ',' |','_|','_ ',' |','_ ',' |','  ','_ ',' |','  ','  ','  ']
    N = len(maze[0,:])
    for i in np.arange(N+1):
        wall_string = ''
        for j in np.arange(N):
            if(i==0):
                if(j==0):
                    wall_string = wall_string+' '
                wall_string = wall_string+'_ '
            else:
                if(j==0):
                    wall_string = wall_string+'|'
                wall_string = wall_string+shapes[int(maze[int(i-1),int(j)])]
        print(wall_string)



# --------- Load matrices for various uses --------- #
Qe = nn.Q_Edges()
Qi = nn.Q_Incomings()
Qc = nn.Q_Connections()
# -------------------------------------------------- #


'''
N = 4
F = [1,1]
rmax = 3
W = 5

maze,Path,Pvalues = mg.Generate_Maze(N)
Walls = mg.Create_Walls_List(maze)
mg.Remove_Walls(Walls,maze,W)
Visualize_Maze(maze)
Dist = nn.Distance(maze,F,Qe,Qc)
QMAT = nn.Initialize(maze,Qe)

Dmax = Walls_Dmax(maze,F,rmax)
for d in np.arange(len(Dmax[0,0,:])):
    print('radius:  ',d+1)
    print(Dmax[:,:,d])
    print('  ')

Pdist = nn.Prob_Dist(QMAT)    
Sf = nn.Failed_Steps(F,maze,rmax,Pdist,Walls_Dmax)

print(Sf)

'''


#------------------------------------------------------------------------------
#  Module 1:    Generating a Perfect Square Maze
#------------------------------------------------------------------------------
'''   # triple ' here
#   User Input:   grid size N  

N = 20

#   Running this code will generate a perfect square maze
#   A visualization of this maze is printed to the console  ( < 1 min )

maze,Path,Pvalues = mg.Generate_Maze(N)
Visualize_Maze(maze)


'''    # triple ' here

#------------------------------------------------------------------------------
#  Module 2:    Generating a Grid with Walls
#------------------------------------------------------------------------------
'''    # triple ' here
#   User Input:   grid size N  |  number of walls W

N = 20
W = 200

#   Running this code will generate a perfect square maze
#   Then, a grid with the desired number of walls is created
#   A visualization of both mazes are printed to the console  ( < 1 min )

maze,Path,Pvalues = mg.Generate_Maze(N)
print('__Initial Perfect Square Maze__')
Visualize_Maze(maze)
Walls = mg.Create_Walls_List(maze)
mg.Remove_Walls(Walls,maze,W)
print('  ')
print('___Grid With ',W,' Walls___')
Visualize_Maze(maze)

'''    # triple ' here


#------------------------------------------------------------------------------
#  Module 3:    Stable Hybrid Search on a Grid with Walls w/ 3D plot
#------------------------------------------------------------------------------
'''   # triple ' here
#   User Input:   grid size N  |  F location  |  number of walls W  |  unitary steps S

N = 30
F = [12,12]
W = 300
S = 120

#   Running this code will generate a grid with the desired number of walls
#   Then, the fastest stable hybrid speed is calculated
#   The results are printed to console
#   A 3D plot of the probability in the system is generated correspondning to the fastest moment ( < 1 min )

maze,Path,Pvalues = mg.Generate_Maze(N)
Walls = mg.Create_Walls_List(maze)
mg.Remove_Walls(Walls,maze,W)
Dist = nn.Distance(maze,F,Qe,Qc)
QMAT = nn.Initialize(maze,Qe)
All_QMAT = np.zeros(shape=(S,N,N,4))
S_Spds = np.zeros(S)
for s in np.arange(S):
    step = int(s+1)
    QMAT = nn.Unitary_Step(QMAT,maze,F,Qc,Qi)
    All_QMAT[s,:,:,:] = QMAT
    Pdist = nn.Prob_Dist(QMAT)
    S_Spds[s] = nn.Stable_Hybrd(Pdist,Dist,step)
S_stp,S_spd = nn.VecMin(S_Spds)

print('_____ N = ',N,'  F = ',F,'  Walls = ',W,' _____')
print('Fastest Stable Hybrid Speed: ',round(S_spd,2))
print('              Unitary Steps: ',round(S_stp+1))

QMAT =  All_QMAT[S_stp,:,:,:]
Pdist = nn.Prob_Dist(QMAT)

X = np.zeros(N**2)
Y = np.zeros(N**2)
Z = np.zeros(N**2)
i = 0
for x in np.arange(N):
    for y in np.arange(N):
        X[i] = int(x+1)
        Y[i] = int(y+1)
        Z[i] = Pdist[x,y]*100
        i = i + 1
bottom = np.zeros_like(Z)
fig = plt.figure(facecolor='white') 
ax = fig.add_subplot(111, projection='3d')       
ax.bar3d(X, Y, bottom, .95, .95, Z, shade=True, color=(.2,.4,.8))


'''    # triple ' here


#------------------------------------------------------------------------------
#  Module 4:    Optimal Hybrid Search on a Grid with Walls
#------------------------------------------------------------------------------
    # triple ' here
#   User Input:   grid size N  |  F location  |  number of walls W  |  max radius rmax  |  unitary steps S

N = 30
F = [12,12]
W = 200
rmax = 10
S = 80

#   Running this code will generate a grid with the desired number of walls
#   Then, the fastest optimal hybrid speed is calculated
#   The results are printed to console
#   A plot of probability accumulated radially is generated ( < 1 min )

maze,Path,Pvalues = mg.Generate_Maze(N)
Walls = mg.Create_Walls_List(maze)
mg.Remove_Walls(Walls,maze,W)
Dist = nn.Distance(maze,F,Qe,Qc)
QMAT = nn.Initialize(maze,Qe)
All_QMAT = np.zeros(shape=(S,N,N,4))
H_Spds = np.zeros(shape=(S,rmax+1))
Dmax = mg.Walls_Dmax(maze,F,rmax,Qe,Qc)
for s in np.arange(S):
    step = int(s+1)
    QMAT = nn.Unitary_Step(QMAT,maze,F,Qc,Qi)
    All_QMAT[s,:,:,:] = QMAT
    Pdist = nn.Prob_Dist(QMAT)
    Pr,Sr = nn.Radial_Info(maze,rmax,F,Pdist,Dist,Qe,Qc)
    Sf = nn.Failed_Steps(F,maze,rmax,Pdist,Dmax)
    H_Spds[s,:] = nn.Hybrid_Speed(Pr,Sr,Sf,step)
best_h,best_r = nn.Best_Hybrid_Spds(H_Spds)

print('_____ N = ',N,'  F = ',F,'  Walls = ',W,' _____')
print('Optimal Radius: ',int(best_r[0]),'  Speed: ',best_r[1],'  Unitary Steps: ',int(best_r[2]))

QMAT =  All_QMAT[int(best_r[2]),:,:,:]
P_r = np.zeros(rmax+1)
Pdist = nn.Prob_Dist(QMAT)
Pr,Sr = nn.Radial_Info(maze,rmax,F,Pdist,Dist,Qe,Qc)
P_r[0] = Pr[0]
for r in np.arange(1,len(Pr)):
    P_r[r] = Pr[r] - Pr[r-1]

Labels = np.arange(rmax+1)
fig = plt.figure(facecolor='white') 
plt.title('N='+str(N)+'  |  Optimal R='+str(int(best_r[0]))+'  |  Radial Probability Around F='+str(F))
plt.bar(Labels[int(best_r[0]+1):rmax+1],P_r[int(best_r[0]+1):rmax+1],align='center',alpha=0.5,color='blue')
plt.bar(Labels[0:int(best_r[0]+1)],P_r[0:int(best_r[0]+1)],align='center',alpha=0.5,color='orange')
plt.text(best_r[0],1.5*(P_r[int(best_r[0])]),'Total Accumulated Probability: '+str(round(Pr[int(best_r[0])],3)),fontsize=8)
plt.xlabel('radius')
plt.ylabel('probability')
plt.xticks(Labels)

    # triple ' here

#------------------------------------------------------------------------------
#  Module 5:    Confirming the Optimal Hybrid Speed Probabilistically
#------------------------------------------------------------------------------
'''    # triple ' here
#   User Input:   grid size N  |  F location  |  number of walls W  |  max radius rmax  |  unitary steps S  |  sample trials T

N = 30
F = [12,12]
W = 200
rmax = 10
S = 90
T = 4*10**4

#   Running this code will generate a grid with the desired number of walls
#   Then, the fastest optimal hybrid speed is calculated
#   Then, using randomly generated measurements, calculates the average optimal hybrid speed  ( 1-2 min )

maze,Path,Pvalues = mg.Generate_Maze(N)
Walls = mg.Create_Walls_List(maze)
mg.Remove_Walls(Walls,maze,W)
Dist = nn.Distance(maze,F,Qe,Qc)
Dmax = mg.Walls_Dmax(maze,F,rmax,Qe,Qc)
QMAT = nn.Initialize(maze,Qe)
All_QMAT = np.zeros(shape=(S,N,N,4))
H_Spds = np.zeros(shape=(S,rmax+1))
for s in np.arange(S):
    step = int(s+1)
    QMAT = nn.Unitary_Step(QMAT,maze,F,Qc,Qi)
    All_QMAT[s,:,:,:] = QMAT
    Pdist = nn.Prob_Dist(QMAT)
    Pr,Sr = nn.Radial_Info(maze,rmax,F,Pdist,Dist,Qe,Qc)
    Sf = nn.Failed_Steps(F,maze,rmax,Pdist,Dmax)
    H_Spds[s,:] = nn.Hybrid_Speed(Pr,Sr,Sf,step)
best_h,best_r = nn.Best_Hybrid_Spds(H_Spds)

QMAT =  All_QMAT[int(best_r[2]),:,:,:]
Pdist = nn.Prob_Dist(QMAT)

print('Optimal Hybrid Radius: ',int(best_r[0]),'     Speed: ',round(best_r[1],1))   

Steps = np.zeros(T)
for t in np.arange(T):
    found = False
    node_x = 0
    node_y = 0
    p_total = 0.0
    steps = 0
    while( found == False ):
        node = nn.Simulate_Measurement(N,Pdist)
        steps = steps + best_r[2]
        r = 0
        Visited   = []
        temp_next = []
        temp_vis  = []
        Next = [[node[0],node[1]]]
        while( r <= best_r[0] ):
            while( len(Next) != 0 ):
                x = Next[0][0]
                y = Next[0][1]
                index = int(maze[x,y])
                steps = steps + 1
                if( [x,y] ==  F ):
                    found = True
                    Steps[t] = steps
                for i in np.arange( int(len(Qe[index])) ):
                    x2 = x + Qc[index][i][0]
                    y2 = y + Qc[index][i][1]
                    new_node = True
                    for j in np.arange( int(len(Visited)) ):
                        if( [x2,y2] == Visited[j] ):
                            new_node = False
                    for jj in np.arange( int(len(temp_next)) ):
                        if( [x2,y2] == temp_next[jj] ):
                            new_node = False
                    if( new_node == True ):
                        temp_next.append([x2,y2])
                temp_vis.append([x,y])
                Next.remove(Next[0])
            for n in np.arange( len( temp_next ) ):
                Next.append( temp_next[n] )
            temp_next = []
            Visited = []
            for v in np.arange( len(temp_vis) ):
                Visited.append(temp_vis[v])
            temp_vis = []
            r = r + 1
    
print('          Probabilitic Average Speed: ',round(( sum(Steps)/T ),1))
              


'''    # triple ' here





