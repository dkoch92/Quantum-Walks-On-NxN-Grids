import NxN_functions as nn
import numpy as np
import math as m
import scipy as sci
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ----------- # ----------- #----------- #----------- #----------- #----------- #----------- #
'''
This python file is designed to showcase several of the results discussed in the paper 
concerning the stable hybrid search algorithm.  The modules are as follows:
    
Module 1: Finding the Fastest Stable Search Speed
Module 2: Plotting Stable Hyrbid Search Speed
Module 3: Radial Probability for Fastest Speed 
Module 4: 3D Plot of Radial Probability
Module 5: Confirming the Stable Hybrid Speed Probabilistically

Each module is a stand-alone block of code, with a marked area for user input
To run a module, remove the triple ' quotes at the beginning and end
Each module produces a result, either printed to console or a plot
'''
# ----------- #----------- #----------- #----------- #----------- #----------- #----------- #

# ------------------------
#      Functions
# ------------------------

def Fastest_Stable(N,F,Qe,Qc,Qi):
    '''
    Input: N (integer)  |  F (length-2 array)
    Calculates the fastest stable hybrid searching speed
    Output: QMAT (N-N-4 matrix)  |  Stp (integer)  |  Dist (N-N matrix)  |  Qm (N-N matrix)
    '''
    S = round( 3.5*( N + m.sqrt(N) ))  
    All_QMAT = np.zeros(shape=(S,N,N,4))
    Qm   = nn.Grid(N)
    QMAT = nn.Initialize(Qm,Qe)
    Dist = nn.Distance(Qm,F,Qe,Qc)
    S_Spds = np.zeros(S)
    for s in np.arange(S):
        step = int(s+1)
        QMAT = nn.Unitary_Step(QMAT,Qm,F,Qc,Qi)
        All_QMAT[s,:,:,:] = QMAT
        Pdist = nn.Prob_Dist(QMAT)
        S_Spds[s] = nn.Stable_Hybrd(Pdist,Dist,step)
    S_stp,S_spd = nn.VecMin(S_Spds)
    Q_best = All_QMAT[S_stp,:,:,:]
    return Q_best,S_stp,Dist,Qm
    


# --------- Load matrices for various uses --------- #
Qe = nn.Q_Edges()
Qi = nn.Q_Incomings()
Qc = nn.Q_Connections()
# -------------------------------------------------- #

#------------------------------------------------------------------------------
#  Module 1:    Finding the Fastest Stable Search Speed
#------------------------------------------------------------------------------
'''   # triple ' here
#   User Input:   grid size N  | F location | unitary steps S

N = 30
F = [12,12]
S = 80

#   Running this code will find the fastest stable hybrid speed
#   Results are printed to console (< 1 min)

Qm   = nn.Grid(N)
QMAT = nn.Initialize(Qm,Qe)
Dist = nn.Distance(Qm,F,Qe,Qc)

S_Spds = np.zeros(S)
for s in np.arange(S):
    step = int(s+1)
    QMAT = nn.Unitary_Step(QMAT,Qm,F,Qc,Qi)
    Pdist = nn.Prob_Dist(QMAT)
    S_Spds[s] = nn.Stable_Hybrd(Pdist,Dist,step)
S_stp,S_spd = nn.VecMin(S_Spds)

print('_____ N = ',N,'  F = ',F,' _____')
print('Fastest Stable Hybrid Speed: ',round(S_spd,2))
print('              Unitary Steps: ',round(S_stp+1))
print('    Average Classical Steps: ',round(S_spd-(S_stp+1),2))


'''    # triple ' here


#------------------------------------------------------------------------------
#  Module 2:    Plotting Stable Hyrbid Search Speed
#------------------------------------------------------------------------------
'''    # triple ' here
#   User Input:   grid size N  | F location | unitary steps S

N = 30
F = [12,12]
S = 180

#   Running this code will gather the stable hybrid speeds for each unitary step
#   A plot is generated (< 1 min)

Qm   = nn.Grid(N)
QMAT = nn.Initialize(Qm,Qe)
Dist = nn.Distance(Qm,F,Qe,Qc)

S_Spds = np.zeros(S)
C_Spds = np.zeros(S)
for s in np.arange(S):
    step = int(s+1)
    QMAT = nn.Unitary_Step(QMAT,Qm,F,Qc,Qi)
    Pdist = nn.Prob_Dist(QMAT)
    S_Spds[s] = nn.Stable_Hybrd(Pdist,Dist,step)
    C_Spds[s] = S_Spds[s] - step
S_stp,S_spd = nn.VecMin(S_Spds)

steps = np.arange(S)
fig = plt.figure(facecolor='white') 
plt.title('N='+str(N)+'  |  F='+str(F)+'  | Classical & Hybrid Speeds')
plt.plot(steps,S_Spds,linewidth=1.2,linestyle='-',color='blue')
plt.plot(steps,C_Spds,linewidth=1,linestyle='--',color='orange')
plt.legend(['Stable Hybrid Speed','Classical Speed'])
plt.xlabel('unitary steps')
plt.ylabel('speed')

'''   # triple ' here


#------------------------------------------------------------------------------
#  Module 3:    Radial Probability for Fastest Speed 
#------------------------------------------------------------------------------
'''   # triple ' here
#   User Input:   grid size N  | F location 

N = 30
F = [12,12]

#   Running this code will first find the fastest stable hybrid speed
#   Then the code calculates the probability accumulated radiialy around F (< 1 min)
#   A plot is generated 

rmax = nn.Max_Radius(N,F)
P_r = np.zeros(rmax+1)
QMAT,S_Stp,Dist,Qm = Fastest_Stable(N,F,Qe,Qc,Qi)
Pdist = nn.Prob_Dist(QMAT)
Pr,Sr = nn.Radial_Info(Qm,rmax,F,Pdist,Dist,Qe,Qc)
P_r[0] = Pr[0]
for r in np.arange(1,len(Pr)):
    P_r[r] = Pr[r] - Pr[r-1]

Labels = np.arange(rmax+1)
fig = plt.figure(facecolor='white') 
plt.title('N='+str(N)+'  |  Radial Probability Around F='+str(F))
plt.bar(Labels,P_r,align='center',alpha=0.5)
plt.xlabel('radius')
plt.ylabel('probability')
plt.xticks(Labels)

'''   # triple ' here


#------------------------------------------------------------------------------
#  Module 4:     3D plot of radial probability
#------------------------------------------------------------------------------
'''    # triple ' here
#   User Input:   grid size N  | F location 

N = 30
F = [12,12]

#   Running this code will first find the fastest stable hybrid speed
#   A 3D plot is generated showing probability of measuring each node (< 1 min)


QMAT,S_Stp,Dist,Qm = Fastest_Stable(N,F,Qe,Qc,Qi)
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
#  Module 5:     Confirming the Stable Hybrid Speed Probabilistically
#------------------------------------------------------------------------------
'''   # triple ' here
#   User Input:   grid size N  | F location  |  sample trials T

N = 30
F = [12,12]
T = 6*10**4

#   Running this code will first find the fastest stable hybrid speed
#   Then, using randomly generated measurements, calculates the average hybrid speed ( 1-2 min )


QMAT,S_Stp,Dist,Qm = Fastest_Stable(N,F,Qe,Qc,Qi)
Pdist = nn.Prob_Dist(QMAT)
S_Spd = nn.Stable_Hybrd(Pdist,Dist,int(S_Stp+1))
print('Fastest Stable Hybrid Speed: ',round(S_Spd,1))

Steps = np.zeros(T)
for t in np.arange(T):
    measurement = sci.rand()
    measured = False
    node_x = 0
    node_y = 0
    p_total = 0.0
    while(measured==False):
        if( p_total < measurement <= (p_total + Pdist[node_x,node_y]) ):
            measured = True
            node = [node_x,node_y]
        else:
            p_total = p_total + Pdist[node_x,node_y]
            node_x = node_x + 1
            if( node_x%N==0 ):
                node_y = node_y + 1
                node_x = 0
    Steps[t] = Dist[node[0],node[1]] + (S_Stp+1)
    
print('Probabilitic Average Speed : ',round(( sum(Steps)/T ),1))
              
'''    # triple ' here            
        
        






    


