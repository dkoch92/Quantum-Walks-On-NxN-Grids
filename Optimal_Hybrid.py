import NxN_functions as nn
import numpy as np
import math as m
import scipy as sci
import matplotlib.pyplot as plt



# ----------- # ----------- #----------- #----------- #----------- #----------- #----------- #
'''
This python file is designed to showcase several of the results discussed in the paper 
concerning the optimal hybrid search algorithm.  The modules are as follows:
    
Module 1: Fastest Optimal Hybrid Speeds for Each Radius
Module 2: Plotting Hybrid Speeds vs Unitary Steps
Module 3: Plotting Hybrid Speeds vs Radius
Module 4: Plotting Radial Probability for Fastest Speed 
Module 5: 3D Plot of Radial Probability
Module 6: Confirming the Stable Hybrid Speed Probabilistically

Each module is a stand-alone block of code, with a marked area for user input.
To run a module, remove the triple ' quotes at the beginning and end.
Each module produces a result, either printed to console or a plot.
When finished with a module, be sure to place the tripl ' quotes back in order to avoid
running multiple modules at once  (and slowing things down)
'''
# ----------- #----------- #----------- #----------- #----------- #----------- #----------- #

# ------------------------
#      Functions
# ------------------------

def Fastest_Hybrid(N,F,Qe,Qc,Qi):
    '''
    Input: N (integer)  |  F (length-2 array)
    Calculates the fastest stable hybrid searching speed
    Output: QMAT (N-N-4 matrix)  |  best_h (rmax-2 matrix)  |  best_r (length-3 array)  |  Dist (N-N matrix)  |  Qm (N-N matrix)
    '''
    S = round( 2.3*( N + m.sqrt(N) ))  
    rmax = round( 3 + 2*N/10 )
    All_QMAT = np.zeros(shape=(S,N,N,4))
    Qm   = nn.Grid(N)
    QMAT = nn.Initialize(Qm,Qe)
    Dist = nn.Distance(Qm,F,Qe,Qc)
    Dmax = nn.Dist_Max(Qm,F,rmax)
    H_Spds = np.zeros(shape=(S,rmax+1))
    for s in np.arange(S):
        step = int(s+1)
        QMAT = nn.Unitary_Step(QMAT,Qm,F,Qc,Qi)
        All_QMAT[s,:,:,:] = QMAT
        Pdist = nn.Prob_Dist(QMAT)
        Pr,Sr = nn.Radial_Info(Qm,rmax,F,Pdist,Dist,Qe,Qc)
        Sf = nn.Failed_Steps(F,Qm,rmax,Pdist,Dmax)
        H_Spds[s,:] = nn.Hybrid_Speed(Pr,Sr,Sf,step)
    best_h,best_r = nn.Best_Hybrid_Spds(H_Spds)
    Q_best = All_QMAT[int(best_r[2]),:,:,:]
    return Q_best,best_h,best_r,Dist,Qm
    

# --------- Load matrices for various uses --------- #
Qe = nn.Q_Edges()
Qi = nn.Q_Incomings()
Qc = nn.Q_Connections()
# -------------------------------------------------- #

#------------------------------------------------------------------------------
#  Module 1:    Module 1: Fastest Hybrid Speeds for Each Radius 
#------------------------------------------------------------------------------
'''  # triple ' here
#   User Input:   grid size N  | F location | unitary steps S  | max radius rmax

N = 30
F = [12,12]
S = 70
rmax = 6

#   Running this code will find the fastest optimal hybrid speeds for each searching radius up to rmax
#   Results are printed to console  ( < 1 min )

Qm   = nn.Grid(N)
QMAT = nn.Initialize(Qm,Qe)
Dist = nn.Distance(Qm,F,Qe,Qc)
Dmax = nn.Dist_Max(Qm,F,rmax)
H_Spds = np.zeros(shape=(S,rmax+1))
for s in np.arange(S):
    step = int(s+1)
    QMAT = nn.Unitary_Step(QMAT,Qm,F,Qc,Qi)
    Pdist = nn.Prob_Dist(QMAT)
    Pr,Sr = nn.Radial_Info(Qm,rmax,F,Pdist,Dist,Qe,Qc)
    Sf = nn.Failed_Steps(F,Qm,rmax,Pdist,Dmax)
    H_Spds[s,:] = nn.Hybrid_Speed(Pr,Sr,Sf,step)
best_h,best_r = nn.Best_Hybrid_Spds(H_Spds)

print('_______________ N = ',N,'  F = ',F,' _______________')
print('  ')
print('__Searching Radius__    __Fastest Speed__    __Unitary Steps__')
for r in np.arange(rmax+1):
    print('        ',r,'                   ',round(best_h[r,1],1),'                ',int(best_h[r,0]))

print('  ')
print('Optimal Radius: ',int(best_r[0]),'  Speed: ',best_r[1],'  Unitary Steps: ',int(best_r[2]))


'''    # triple ' here

#------------------------------------------------------------------------------
#  Module 2:    Plotting Hybrid Speeds vs Unitary Steps
#------------------------------------------------------------------------------
'''   # triple ' here
N = 30
F = [12,12]
S = 100
rmax = 10

#   Running this code will gather the stable hybrid speeds for each unitary step
#   A plot is generated (< 1 min)

Qm   = nn.Grid(N)
QMAT = nn.Initialize(Qm,Qe)
Dist = nn.Distance(Qm,F,Qe,Qc)
Dmax = nn.Dist_Max(Qm,F,rmax)
H_Spds = np.zeros(shape=(S,rmax+1))
C_Spds = np.zeros(S)

for s in np.arange(S):
    step = int(s+1)
    QMAT = nn.Unitary_Step(QMAT,Qm,F,Qc,Qi)
    Pdist = nn.Prob_Dist(QMAT)
    Pr,Sr = nn.Radial_Info(Qm,rmax,F,Pdist,Dist,Qe,Qc)
    Sf = nn.Failed_Steps(F,Qm,rmax,Pdist,Dmax)
    H_Spds[s,:] = nn.Hybrid_Speed(Pr,Sr,Sf,step)
best_h,best_r = nn.Best_Hybrid_Spds(H_Spds)

O_Spds = H_Spds[:,int(best_r[0])]
for s2 in np.arange(S):
    C_Spds[s2] = O_Spds[s2] - int(s2+1)

steps = np.arange(S)
fig = plt.figure(facecolor='white') 
plt.title('N='+str(N)+'  |  F='+str(F)+'  | Searching Radius: '+str(int(best_r[0]))+'  ')
plt.plot(steps,O_Spds,linewidth=1.2,linestyle='-',color='blue')
plt.plot(steps,C_Spds,linewidth=1,linestyle='--',color='orange')
plt.scatter(best_r[2],best_r[1],s=110,marker='*',color='purple')
plt.legend(['Optimal Hybrid Speed','Classical Speed','Fastest Hyrbid Speed'],loc='upper left')
plt.axis([1,S,0,round(4.5*best_r[1])])
plt.xlabel('unitary steps')
plt.ylabel('speed')


'''   # triple ' here


#------------------------------------------------------------------------------
#  Module 3:    Plotting Hybrid Speeds vs Radius
#------------------------------------------------------------------------------
'''    # triple ' here
#   User Input:   grid size N  | F location | unitary steps S

N = 30
F = [12,12]
S = 60

#   Running this code will find the fastest optimal hybrid speeds for each searching radius
#   A plot is generated  ( ~ 1 min)

S = round( 2.2*(N + m.sqrt(N)) )
rmax = round( .75*nn.Max_Radius(N,F) )
Qm   = nn.Grid(N)
QMAT = nn.Initialize(Qm,Qe)
Dist = nn.Distance(Qm,F,Qe,Qc)
Dmax = nn.Dist_Max(Qm,F,rmax)
H_Spds = np.zeros(shape=(S,rmax+1))
for s in np.arange(S):
    step = int(s+1)
    QMAT = nn.Unitary_Step(QMAT,Qm,F,Qc,Qi)
    Pdist = nn.Prob_Dist(QMAT)
    Pr,Sr = nn.Radial_Info(Qm,rmax,F,Pdist,Dist,Qe,Qc)
    Sf = nn.Failed_Steps(F,Qm,rmax,Pdist,Dmax)
    H_Spds[s,:] = nn.Hybrid_Speed(Pr,Sr,Sf,step)
best_h,best_r = nn.Best_Hybrid_Spds(H_Spds)

rvec = np.arange(rmax+1)
fig = plt.figure(facecolor='white') 
plt.title('N='+str(N)+'  |  F='+str(F)+'  | Fastest Hybrid Speeds')
plt.scatter(rvec,best_h[:,1],s=6,color='black',marker='o')
plt.plot(rvec,best_h[:,1],linewidth=.4,linestyle='-',color='blue')
plt.scatter(best_r[0],best_r[1],s=110,color='purple',marker='*')
plt.xlabel('searching radius')
plt.ylabel('speed')

'''   # triple ' here

#------------------------------------------------------------------------------
#  Module 4:    Plotting Radial Probability for Fastest Speed 
#------------------------------------------------------------------------------
'''   # triple ' here
#   User Input:   grid size N  | F location 

N = 30
F = [12,12]

#   Running this code will first find the fastest optimal hybrid speed
#   Then the code calculates the probability accumulated radialy around F
#   A plot is generated  ( < 1 min )


QMAT,best_h,best_r,Dist,Qm = Fastest_Hybrid(N,F,Qe,Qc,Qi)
rmax = round( 0.75 * nn.Max_Radius(N,F) )
P_r = np.zeros(rmax+1)
Pdist = nn.Prob_Dist(QMAT)
Pr,Sr = nn.Radial_Info(Qm,rmax,F,Pdist,Dist,Qe,Qc)
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


'''    # triple ' here


#------------------------------------------------------------------------------
#  Module 5:     Confirming the Optimal Radius Hybrid Speed Probabilistically
#------------------------------------------------------------------------------
'''    # triple ' here
#   User Input:   grid size N  | F location  |  sample trials T

N = 30
F = [12,12]
T = 4*10**4

#   Running this code will first find the fastest stable hybrid speed
#   Then, using randomly generated measurements, calculates the average hybrid speed  ( 1-2 min )

QMAT,best_h,best_r,Dist,Qm = Fastest_Hybrid(N,F,Qe,Qc,Qi)    
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
                index = int(Qm[x,y])
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
        
    


