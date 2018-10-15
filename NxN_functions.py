import numpy as np
import math as m
import scipy as sci



# ----------- # ----------- #----------- #----------- #----------- #----------- #----------- #
'''
This python file conatins various defined functions, meant to be imported and used by other python files.
Make sure that this file is in the same working directory as all of the other python files.
The codes that import from this file are the following:

Test_Import.py
Optimal_Hybrid.py
Stable_Hybrid.py
Walls_Search.py
Lattice.py

Running this code produces no results, and I do not reccommend making any alterations to functions
in this file, as it may cause other python files to run improperly.

To check that this file in the correct direcorty, please run Test_Import.py and make sure the results
that print to the console confirm all functions have been importing properly
'''
# ----------- #----------- #----------- #----------- #----------- #----------- #----------- #

# ------------------------
#      Functions
# ------------------------


def Initialize(maze,Qe):
    '''
    Creates the initial superposition of all states in the system for a grid
    Returns a NxNx4 matrix
    '''
    N = len(maze[:,0])
    total_states = 0
    sides = [0,1,1,1,1,2,2,2,2,2,2,3,3,3,3,4]
    for x in np.arange(N):
        for y in np.arange(N):
            total_states = total_states + sides[ int(maze[x,y]) ]
    amp = 1.0/m.sqrt(total_states)
    Qmat = np.zeros(shape=(N,N,4))
    for i in np.arange(N):
        for j in np.arange(N):
            for k in np.arange(len(Qe[int(maze[i,j])])):
                Qmat[i,j,Qe[int(maze[i,j])][k]] = amp
    return Qmat

def Grid(N):
    '''
    Creates a matrix representing all of the nodes' indeices in the system
    Returns an NxN matrix Maze
    '''
    maze = np.zeros(shape=(N,N))
    maze[0,0] = 10
    maze[0,N-1] = 7
    maze[N-1,N-1] = 5
    maze[N-1,0] = 8
    for a in np.arange(1,N-1):
        maze[0,a] = 13
        maze[a,N-1] = 12
        maze[N-1,a] = 11
        maze[a,0] = 14
        for b in np.arange(1,N-1):
            maze[a,b] = 15
    return maze


def Q_Edges():
    '''
    Returns the matrix Q_edges, which contains info about edge locations
    '''
    Q_e = [[],[0],[1],[2],[3],[0,1],[0,2],[0,3],[1,2],[1,3],[2,3],[0,1,2],[0,1,3],[0,2,3],[1,2,3],[0,1,2,3]]
    return Q_e

def Q_Connections():
    '''
    Returns the matrix Q_connections, which contains info about neighboring states
    '''
    Q_c = [ [[]],[[0,-1,2]],[[-1,0,3]],[[0,1,0]],[[1,0,1]],
            [[0,-1,2],[-1,0,3]],[[0,-1,2],[0,1,0]],[[0,-1,2],[1,0,1]],[[-1,0,3],[0,1,0]],[[-1,0,3],[1,0,1]],[[0,1,0],[1,0,1]],
            [[0,-1,2],[-1,0,3],[0,1,0]],[[0,-1,2],[-1,0,3],[1,0,1]],[[0,-1,2],[0,1,0],[1,0,1]],[[-1,0,3],[0,1,0],[1,0,1]],
            [[0,-1,2],[-1,0,3],[0,1,0],[1,0,1]] ]
    return Q_c

def Q_Incomings():
    '''
    Returns the matrix Q_incomings, which contains info about the node's connections for the U operation
    '''
    Q_i = [  [[]],[[0]],[[1]],[[2]],[[3]],
             [[0,1],[1,0]],[[0,2],[2,0]],[[0,3],[3,0]],[[1,2],[2,1]],[[1,3],[3,1]],[[2,3],[3,2]],
             [[0,1,2],[1,0,2],[2,0,1]],[[0,1,3],[1,0,3],[3,0,1]],[[0,2,3],[2,0,3],[3,0,2]],[[1,2,3],[2,1,3],[3,1,2]],
             [[0,1,2,3],[1,0,2,3],[2,0,1,3],[3,0,1,2]] ]
    return Q_i
    

def Coefficients(index):
    '''
    Returns an array containing the coefficients for a given index type
    '''
    C=[[1.0],[0,1.0],[-1.0/3,2.0/3,2.0/3],[-.5,.5,.5,.5]]
    I = [99,0,0,0,0,1,1,1,1,1,1,2,2,2,2,3]
    coef = C[ int(I[ int(index) ]) ]
    return coef


def Unitary_Step(Qmat,maze,f,Qc,Qi):
    '''
    Performs 1 unitary step on the Quantum System
    '''
    N = len( maze[0,:] )
    Qf = np.zeros(shape=(N,N,4))       #creates an empty matrix to hold all the final values
    Q = np.zeros(shape=(N,N,4))        #creates an empty matrix that will copy all the values from Mat
    Q = Qmat[:,:]
    for x in np.arange(N):
        for y in np.arange(N):
            shape_index = int(maze[x,y])
            C = Coefficients(shape_index)
            if( [x,y] == f ):
                for k in np.arange( len(C) ):
                    C[k] = C[k]*(-1.0)
            for i in np.arange( len(C) ):
                xf = x + Qc[shape_index][i][0]
                yf = y + Qc[shape_index][i][1]
                zf = Qc[shape_index][i][2]
                amp = 0
                for j in np.arange( len(C) ):
                    z = Qi[shape_index][i][j]
                    amp = amp + C[j]*Q[x,y,z]
                Qf[xf,yf,zf] = amp
    return Qf

def Unitary_Check(Qmat,maze):
    '''
    Checks to make sure that all of the probability in the system sums to 1
    Prints a message if the system is not unitary
    '''
    N = len( maze[0,:] )
    Qe = Q_Edges()
    prob = 0
    for x in np.arange(N):
        for y in np.arange(N):
            index = int( maze[x,y] )
            for i in np.arange( len( Qe[index] ) ):
                prob = prob + ( Qmat[x,y,Qe[index][i]] )**2
    if( round(prob,6) != 1.0 ):
        print( '_____NOT UNITARY_____' )
        print( 'Total Probability:',prob )

def Distance(maze,F,Qe,Qc):
    '''
    Calculates the nmumber of classical steps to a location F, from any point [x,y]
    Returns the matrix Dist
    '''
    N = len(maze[:,0])
    D = np.zeros(shape=(N,N))
    for X in np.arange(N):
        #print('Dist X = ',X)
        for Y in np.arange(N):
            Next      = [[int(X),int(Y)]]
            Visited   = []
            temp_next = []
            temp_vis  = []
            steps = -1
            found_F = False
            while( found_F == False ):
                while( len(Next) != 0 ):
                    x = Next[0][0]
                    y = Next[0][1]
                    index = int(maze[x,y])
                    steps = steps + 1
                    if( [x,y] ==  F ):
                        found_F = True
                        D[X,Y] = steps
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
    return D

def Failed_Steps(F,maze,rmax,Pdist,Dmax):
    '''
    Calculates the average number of steps for a failed trial
    '''
    N = len(Pdist[:,0])
    Sr = np.zeros(rmax+1)
    for r in np.arange(rmax):
        avg = 0
        total_p = 0
        for X in np.arange(N):
            for Y in np.arange(N):
                if(Dmax[X,Y,r] != 0):
                    avg = avg + Pdist[X,Y]*Dmax[X,Y,r]
                    total_p = total_p + Pdist[X,Y]
        Sr[r+1] = avg/total_p
    return Sr
        
                    
                    

def Dist_Max(Qm,F,rmax):
    '''
    Calculates the maximum number of steps for a failed trial, for all [x,y] and r values
    Returns Dmax
    '''
    N = len(Qm[0,:])
    Dm = np.zeros(shape=(N,N,int(rmax)))
    for i in np.arange(rmax):
        #print('Dmax Radius:   ',i)
        r = int(i+1)
        L = []
        for j in np.arange(-r,r+1):
            x = int(j)
            y = int(r-abs(x))
            L.append([x,y])
            if(y!=0):
                L.append([x,-y])
        total_nodes = 0
        for X in np.arange(N):
            for Y in np.arange(N):
                if( ((abs(X-F[0]) + abs(Y-F[1])) > r) and (0<=X<N) and (0<=Y<N) ):
                    total_nodes = total_nodes + 1
                    Dm[X,Y,i] = Max_Steps(N,L,[X,Y])
    return Dm
                    
        
    
def Max_Steps(N,nodes,loc):
    '''
    Calculates the maximum number of steps for a failed trial, from a location [x,y]
    ''' 
    OB = 0
    for n in np.arange(len(nodes)):
        x = loc[0] + nodes[n][0]
        y = loc[1] + nodes[n][1]
        if( y < 0 ):
            OB = OB + abs(y)
        if( y >= N ):
            OB = OB + y - (N-1)
        if( (0<=y<N) and (x < 0) ):
            OB = OB + abs(x)
        if( (0<=y<N) and (x >= N) ):
            OB = OB + x - (N-1)
    r = abs(nodes[0][0])
    s = 2*r*(r+1) - OB
    return s
        

def Max_Radius(N,F):
    '''
    Input: N (integer)  |  F (length-2 array)
    Calculates the maximum searching radius
    Output: rmax (integer)
    '''
    C = [[0,0],[0,N-1],[N-1,N-1],[N-1,0]]
    D = [0,0,0,0]
    for c in np.arange(4):
        D[c] = abs(F[0]-C[c][0]) + abs(F[1]-C[c][1])
    d = int( max(D) )
    return d

def Prob_Dist(qmat):
    '''
    Takes in QMAT
    Calculates the probability of measuring a given node
    Returns an NxN matrix
    '''
    N = len(qmat[:,0,0])
    P = np.zeros(shape=(N,N))
    for x in np.arange(N):
        for y in np.arange(N):
            prob = 0
            for e in np.arange(4):
                prob = prob + qmat[x,y,e]**2    
            P[x,y] = prob
    return P

def Stable_Hybrd(P,D,step):
    '''
    Takes in Prob_Dist, Dist, and step
    Calculatves the Stable Hybrid Speed
    Returns a float
    '''
    N = len(D[0,:])
    spd = 0
    for x in np.arange(N):
        for y in np.arange(N):
            spd = spd + P[x,y]*D[x,y]
    spd = round((spd + step),5)
    return spd

def Avg_Steps_R(P,D,r,F):
    '''
    Takes in Prob_Dist, Dist, and radius
    Calculatves the average number of steps to find F within a radius r (successful trial)
    Returns a float
    '''
    spd = 0
    total_P = 0
    for x in np.arange(F[0]-r,F[0]+r+1):
        for y in np.arange(F[0]-r,F[0]+r+1):
            if( (abs(F[0]-x) + abs(F[1]-y)) <= r ):
                spd = spd + P[x,y]*D[x,y]
                total_P = total_P + P[x,y]
    spd = round((spd/total_P),5)
    return spd

def Radial_Info(maze,rmax,F,P,Dist,Qe,Qc):
    '''
    Takes in ...
    Gathers all relevant info for speed calcultaions in one breadth-first walk
    Returns...
    '''
    Next      = [F]
    Visited   = []
    temp_next = []
    temp_vis  = []
    N = len( maze[0,:] )
    P_all = np.zeros(N**2)
    S_all = np.zeros(N**2)
    Pr = np.zeros(rmax+1)
    Sr = np.zeros(rmax+1)
    nn = -1
    for r in np.arange(rmax+1):
        while( len(Next) != 0 ):
            x = Next[0][0]
            y = Next[0][1]
            nn = nn + 1
            P_all[nn]   = P[x,y]
            S_all[nn] = Dist[x,y]
            index = int(maze[x,y])
            for i in np.arange( int(len(Qe[index])) ):
                x2 = x + Qc[index][i][0]
                y2 = y + Qc[index][i][1]
                new_node = True
                for j in np.arange( int(len(Visited)) ):
                    if( [x2,y2] == Visited[j] ):
                        new_node = False
                for jj in np.arange(len(temp_next)):
                    if([x2,y2] == temp_next[jj] ):
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
        Sc = 0
        for s in np.arange( nn+1 ):
            Sc = Sc + S_all[s]*P_all[s]
        Pr[r] = sum(P_all)
        Sr[r] = Sc/sum(P_all)
    return Pr,Sr

def Hybrid_Speed(Pr,Sr,Sf,step):
    '''
    Calculates the Hybrid Search Speed for all radius up to rmax
    Returns an array H_Spd containing hybrid speeds
    '''
    rmax = len(Pr)
    H_Spd = np.zeros(rmax)
    Us = int(step)
    for r in np.arange(rmax):
        x = 1.0 - Pr[r]
        A = Us + Sf[r]*1.0
        B = Us + Sr[r]*1.0
        H_Spd[r] = (1.0-x)*( A*x**2 - B*x**2 + B*x  )/( x*(1-x)**2 )
    return H_Spd

def VecMin(vec):
    '''
    Finds the location and value of the lowest value in a vector
    '''
    value = min(vec)
    for i in np.arange( len(vec) ):
        if( vec[i] ==  value ):
            place = i
    return place,value

def Best_Hybrid_Spds(H_spd):
    '''
    Calculates the fastest hybrid speeds for each radius
    '''
    rmax = len(H_spd[0,:])
    best_spds = np.zeros(shape=(rmax,2))
    best_best = np.zeros(3)
    for r in np.arange(rmax):
        best_spds[r,0],best_spds[r,1] = VecMin( H_spd[:,r] )
    best_best[0],best_best[1] = VecMin( best_spds[0:rmax-1,1] )
    best_best[1] = round(best_best[1],2)
    best_best[2] = best_spds[int(best_best[0]),0]
    return best_spds,best_best


def Simulate_Measurement(N,Pdist):
    '''
    Input: N (integer)  |  Pdist (N-N matrix)
    Simulates a measurment on the quantum system, yielding a node
    Output: node (length-2 array)
    '''
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
    return node

#--------------
#--------------
    
