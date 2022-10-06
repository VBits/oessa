"""
Use this script in order to merge pickle files and train a neural network on multiple mice
"""
import os
from natsort import natsorted
import glob
import pandas as pd
import numpy as np

def report_states(EphysDir,Folder):
    os.chdir(EphysDir+Folder)
    print ('testing how many states per mouse')
    for counter, file in enumerate(natsorted(glob.glob("states_*.pkl"))):
        print(file)
        states_file = pd.read_pickle(file)
        print(np.unique(states_file['states'],return_counts=True))

def combine_files(EphysDir,Folder,load=True):
    # Generate a file to be used for models while avoiding overfitting
    os.chdir(EphysDir+Folder)
    if load:
        print ('loading file with states for all the mice')
        Sxx_combined = pd.read_pickle(EphysDir + Folder + 'All_mice_Sxx_combined.pkl')
        states_combined = pd.read_pickle(EphysDir + Folder + 'All_mice_states_combined.pkl')
        multitaper_combined = pd.read_pickle(EphysDir + Folder + 'All_mice_multitaper_combined.pkl')
    else:
        print ('saving file with states for all the mice')
        Sxx_combined = pd.DataFrame()
        states_combined = pd.DataFrame()
        multitaper_combined = pd.DataFrame()
        for counter, file in enumerate(natsorted(glob.glob("Sxx*.pkl"))):
            print(file)
            Sxx_file = pd.read_pickle(file)
            file_id = file.split("_",2)[-1]
            states_file = pd.read_pickle('states_{}'.format(file_id))
            multitaper_file = pd.read_pickle('Multitaper_df_{}'.format(file_id))
            print(len(Sxx_file),len(multitaper_file),len(states_file))
            # #trim the Sxx file to match the states file length
            # Sxx_file = Sxx_file[:len(states_file)]
            #process with log 10
            # multitaper_file = 10*np.log(multitaper_file[:len(states_file)])
            Sxx_combined = Sxx_combined.append(Sxx_file,ignore_index=True)
            states_combined = states_combined.append(states_file, ignore_index=True)
            multitaper_combined = multitaper_combined.append(multitaper_file, ignore_index=True)
        Sxx_combined.to_pickle(EphysDir + Folder + 'All_mice_Sxx_combined.pkl')
        multitaper_combined.to_pickle(EphysDir + Folder + 'All_mice_multitaper_combined.pkl')
        states_combined.to_pickle(EphysDir + Folder + 'All_mice_states_combined.pkl')
    return Sxx_combined, multitaper_combined, states_combined


EphysDir = 'D:/Project_Mouse/Ongoing_analysis/'
Folder = 'Avoid_overfitting/'
#Optional: Check if any datasets have 3 states and get rid of them
report_states(EphysDir,Folder)
#combine files to use for the training
Sxx_combined, multitaper_combined,states_combined = combine_files(EphysDir,Folder,load=True)

# get rid of any states that are artifacts
Sxx_combined = Sxx_combined[states_combined['states']!='ambiguous']
multitaper_combined = multitaper_combined[states_combined['states']!='ambiguous']
states_combined = states_combined[states_combined['states']!='ambiguous']

rand_idx = get_random_idx(Sxx_combined,size=200000)
m.state_df = states_combined.loc[rand_idx]
m.Sxx_df = Sxx_combined.loc[rand_idx]
m.multitaper_df = multitaper_combined.loc[rand_idx]

############################################################



