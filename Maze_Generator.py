import numpy as np
import math as m
import scipy as sci

# ----------- # ----------- #----------- #----------- #----------- #----------- #----------- #
'''
This python file conatins various defined functions, meant to be imported and used by other python files.
Make sure that this file is in the same working directory as all of the other python files.
The codes that import from this file are the following:

Walls_Search.py

Running this code produces no results, and I do not reccommend making any alterations to functions
in this file, as it may cause other python files to run improperly.

To check that this file in the correct direcorty, please run Test_Import.py and make sure the results
that print to the console confirm all functions have been importing properly
'''
# ----------- #----------- #----------- #----------- #----------- #----------- #----------- #

def Initialize_Maze(n):
    '''
    Creates all the initial arrays and matrix for the maze generating algorithm
    '''
    maze = np.zeros(shape=(n,n))
    for i in np.arange(n):
        for j in np.arange(n):
            maze[i,j] = 20
    visited = []
    stack   = []
    location = [0,0]
    return maze,location,visited,stack
    
def Check_Moves(location,visited,n):
    '''
    Check all available locations and determines which ones are suitable for Move
    '''
    possibles = []
    candidate = [0,0]
    candidate_list = [[0,0],[0,0],[0,0],[0,0]]
    directions = [[-1,0],[0,-1],[1,0],[0,1]]
    for i in np.arange(4):
        moveable_bool = True
        candidate[0] = location[0] + directions[i][0]
        candidate[1] = location[1] + directions[i][1]
        candidate_list[i][0] = candidate[0]
        candidate_list[i][1] = candidate[1]
        if( (candidate[0] < 0) or (candidate[1] < 0) or (candidate[0] >= n) or (candidate[1] >= n) ):
            check_visited = False
            moveable_bool = False
        else:
            check_visited = True
        check_index = -1
        while(check_visited):
            check_index = check_index + 1
            if( candidate == visited[check_index] ):
                check_visited = False
                moveable_bool = False
            if( (check_index + 1) == len(visited) ):
                check_visited = False
        if( moveable_bool ):
            possibles.append( candidate_list[i] )
    return possibles
    
def Pick_Move(moveables):
    '''
    Picks a random direction to move
    '''
    options = int(len(moveables))
    roll = round( options*sci.rand() - .499999999 )
    next_direction = moveables[roll]
    return next_direction
    
def Location_Value(maze,location,next_move):
    '''
    Function used by Move that assign numerical values representing connections
    '''
    previous_value = maze[location[0],location[1]]
    values_table = np.array([[21,22,23,24],[99,25,26,27],[25,99,28,29],[26,28,99,30],[27,29,30,99],
                             [99,99,31,32],[99,31,99,33],[99,32,33,99],[31,99,99,34],[32,99,34,99],
                             [33,34,99,99],[99,99,99,35],[99,99,35,99],[99,35,99,99],[35,99,99,99]])
    direction,reverse = Determine_Direction(location,next_move)
    location_value = values_table[int(previous_value-20),int(direction)]
    next_value = values_table[0,int(reverse)]
    maze[location[0],location[1]]   = location_value
    maze[next_move[0],next_move[1]] = next_value
    

def Determine_Direction(current,move):
    '''
    Function used by Location_Value in order to properly assign values to locations in maze
    '''
    V = move[0] - current[0]
    H = move[1] - current[1]
    if( (V==0) and (H==-1) ):
        direction = 0
        reverse = 2
    if( (V==-1) and (H==0) ):
        direction = 1
        reverse = 3
    if( (V==0) and (H==1) ):
        direction = 2
        reverse = 0
    if( (V==1) and (H==0) ):
        direction = 3
        reverse = 1
    return direction,reverse



def Determine_Next_Location(maze,location,visited,stack,path,F):
    '''
    Determine the next NEW location, or else backtrack
    '''
    n = len( maze[0,:] )
    next_bool = False
    end_bool = False
    once_add = True
    while( next_bool == False ):
        if(location == [n-1,n-1]):
            visited.append(location)
            for i in np.arange( len(stack) ):
                path.append( stack[i] )
            path.append( [n-1,n-1] )
            once_add = False
            location = stack[-1]
            stack.remove(location)
        else:
            possibles = Check_Moves(location,visited,n)
            if( len(possibles) == 0 ):
                if(once_add):
                    visited.append(location)
                    once_add = False
                if( len(stack) == 0 ):
                    next_bool = True
                    end_bool = True
                    next_move = [-1,-1]
                else:
                    location = stack[-1]
                    stack.remove(location)
            else:
                next_move = Pick_Move(possibles)
                next_bool = True
    return end_bool,next_move,location,stack,path
    
 
def First_Move(maze,location,visited,stack):
    possibles = [[1,0],[0,1]]
    Next = Pick_Move(possibles)
    if( Next == [1,0] ):
        maze[0,0] = 24
        maze[1,0] = 22
    if( Next == [0,1] ):
        maze[0,0] = 23
        maze[0,1] = 21
    visited.append([0,0])
    stack.append([0,0])
    location = Next
    return location

def Move(maze,next_move,location,visited,stack):
    '''
    Function that advances the maze generating algorithm 1 node, keeping track of all relevant information
    '''
    next_temp = next_move
    Location_Value(maze,location,next_move)
    visited.append(location)
    stack.append(location)
    location = next_temp
    return location
    
def Clean_Maze(maze):
    '''
    
    '''
    n = len( maze[0,:] )
    for i in np.arange(n):
        for j in np.arange(n):
            maze[i,j] = int( maze[i,j] - 20 )
    
def Clean_Path(maze,path):
    '''
    Figures out which edges for 3 and 4-connections are the correct path
    '''
    length = len( path )
    path_values = np.zeros( length )
    X  = [-1,1]
    N  = [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
    N2 = [5,6,7,8,9,10]
    for i in np.arange( length ):
        x = path[i][0]
        y = path[i][1]
        if( maze[x,y] <= 10 ):
            path_values[i] = maze[x,y]
        else:
            neighbors = [0,0]
            for j in np.arange(2):
                x2 = path[i+X[j]][0]
                y2 = path[i+X[j]][1]
                if( (x2 == x) and (y2 == (y-1) ) ):
                    neighbors[j] = 1
                if( (x2 == (x-1) ) and (y2 == y ) ):
                    neighbors[j] = 2
                if( (x2 == x) and (y2 == (y+1) ) ):
                    neighbors[j] = 3
                if( (x2 == (x+1) ) and (y2 == y ) ):
                    neighbors[j] = 4
            for k in np.arange( 6 ):
                if( neighbors == N[k] ):
                    path_values[i] = N2[k]
    return path_values
            

def Generate_Maze(size):
    '''
    
    '''
    Maze,Location,Visited,Stack = Initialize_Maze(size)
    Location = First_Move(Maze,Location,Visited,Stack)
    Path = []
    FINISH = False
    F_bool = False
    while(FINISH == False):
        FINISH,Next_Location,Location,Stack,Path = Determine_Next_Location(Maze,Location,Visited,Stack,Path,F_bool)
        if(FINISH != True):
            Location = Move(Maze,Next_Location,Location,Visited,Stack)    
    Clean_Maze(Maze)
    Pvalues = Clean_Path(Maze,Path)
    return Maze,Path,Pvalues
    

def Maze(N):
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

def Walls_Dmax(maze,F,rmax,Qe,Qc):
    '''
    Calculates the maximum number of steps for a failed trial, for all [x,y] and r values
    Returns Dmax
    '''
    N = len(maze[0,:])
    Dm = np.zeros(shape=(N,N,int(rmax)))
    for X in np.arange(N):
        for Y in np.arange(N):
            Next      = [[X,Y]]
            Visited   = []
            temp_next = []
            temp_vis  = []
            steps = -1
            found_F = False
            for r in np.arange(rmax+1):
                if(found_F == False):
                    while( len(Next) != 0 ):
                        x = Next[0][0]
                        y = Next[0][1]
                        if( [x,y] == F ):
                            found_F = True
                        steps = steps + 1
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
                if(found_F == True):
                    if(r >= 1):
                        Dm[X,Y,int(r-1)] = 0
                else:
                    if(r >= 1):
                        Dm[X,Y,int(r-1)] = steps
    return Dm
                    


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

def Q_Walls():
    '''
    Returns the matrix Q_Walls, which contains info about the node's connections for the U operation
    '''
    Q_w = [[],[],[],[],[],
           [2,1,0,0],[3,0,1,0],[4,0,0,1],[0,3,2,0],[0,4,0,2],[0,0,4,3],
           [8,6,5,0],[9,7,0,5],[10,0,7,6],[0,10,9,8],[14,13,12,11]] 
    return Q_w
    

def Coefficients(index):
    '''
    Returns an array containing the coefficients for a given index type
    '''
    C=[[1.0],[0,1.0],[-1.0/3,2.0/3,2.0/3],[-.5,.5,.5,.5]]
    I = [99,0,0,0,0,1,1,1,1,1,1,2,2,2,2,3]
    coef = C[ int(I[ int(index) ]) ]
    return coef


def Create_Walls_List(maze):
    '''
    Goes through the matrix maxe[x,y] and creates a list of all wall locations
    Returns a list
    '''
    Walls = []
    N = len(maze[0,:])
    W = [[],[1],[0],[0,1],[0,1],[],[1],[1],[0],[0],[0,1],[],[],[1],[0],[]]
    for x in np.arange(N):
        for y in np.arange(N):
            index = int( maze[x,y] )
            for i in np.arange( len(W[index]) ):
                Walls.append( [x,y,W[index][i]] )
    for x2 in np.arange(N):
        Walls.remove( [x2,0,0] )
    for y2 in np.arange(N):
        Walls.remove( [0,y2,1] )
    return Walls

def Remove_Walls(Walls,maze,total):
    '''
    Removes walla from the matrix maze, and updates it
    Keeps removing walls until reaching a desired number, total
    Returns maze
    '''
    destroy = int( len(Walls) - total )
    W = [[0,0,0,0],[0,5,6,7],[5,0,8,9],[6,8,0,10],[7,9,10,0],[0,0,11,12],[0,11,0,13],[0,12,13,0],
         [11,0,0,14],[12,0,14,0],[13,14,0,0],[0,0,0,15],[0,0,15,0],[0,15,0,0],[15,0,0,0],[0,0,0,0]]
    for w in np.arange( destroy ):
        wall = int( m.floor(len(Walls)*sci.rand()) )
        x = int( Walls[wall][0] )
        y = int( Walls[wall][1] )
        e = int( Walls[wall][2] )
        index = int(maze[x,y])
        if( e == 0 ):
            x2 = int(x)
            y2 = int(y - 1)
            e2 = 2
        if( e == 1 ):
            x2 = int(x - 1)
            y2 = int(y)
            e2 = 3
        index2 = int(maze[x2,y2])
        maze[x,y]   = W[index][e]
        maze[x2,y2] = W[index2][e2]
        Walls.remove(Walls[wall])
  
