import NxN_functions as nn
import Maze_Generator as mg
import numpy as np


# ----------- # ----------- #----------- #----------- #----------- #----------- #----------- #
'''
This python file is designed to make sure all files import correctly.
To ensure that all of the code assembled here runs as intended, please make sure
that all of the python files download from the github project are located within
the same working directory.

Running this code will make sure that all functions import properly, and return a
messege if so.

If the messege at the end of this code does not display that everything is running
correctly, please consider reloacting all the files, or perhaps redownloading the 
project again and manually checking that all the files are in the same foler.

'''
# ----------- #----------- #----------- #----------- #----------- #----------- #----------- #

List_of_nn_functions = ['Initialize','Grid','Q_Edges','Q_Connections','Q_Incomings','Coefficients',
                        'Unitary_Step','Unitary_Check','Distance','Failed_Steps','Dist_Max','Max_Steps',
                        'Max_Radius','Prob_Dist','Stable_Hybrd','Avg_Steps_R','Radial_Info',
                        'Hybrid_Speed','VecMin','Best_Hybrid_Spds','Simulate_Measurement']

List_of_mg_functions = ['Initialize_Maze','Check_Moves','Pick_Move','Location_Value','Determine_Direction',
                        'Determine_Next_Location','First_Move','Move','Clean_Maze','Clean_Path',
                        'Generate_Maze','Maze','Walls_Dmax','Q_Walls','Create_Walls_List','Remove_Walls']

#----------------------------------------------------------------------

nn_all_bool = True
for s in np.arange(len(List_of_nn_functions)):
    func = 'nn.'+List_of_nn_functions[s]
    if( eval( func ) ):
        print('imported -- '+func)
    else:
        nn_all_bool = False
        
if(nn_all_bool):
    print('     ------------      ')
    print('All NxN_functions imported properly')
    print('  ')
    
#----------------------------------------------------------------------

mg_all_bool = True
for s in np.arange(len(List_of_mg_functions)):
    func = 'mg.'+List_of_mg_functions[s]
    if( eval( func ) ):
        print('imported -- '+func)
    else:
        mg_all_bool = False
        
if(mg_all_bool):
    print('     ------------      ')
    print('All Maze_Generator functions imported properly')