"""
experimental_path starts with an experiment ID
file starts with a file ID
We use dates that the experiment first started and a file first created as unique IDs
"""
#Update for each mouse
base_path = 'D:/Project_Mouse/Ongoing_analysis/'
experimental_path = '201022_Nms_NO_misting_burrowing/Ephys/'
file = '210120_000_SD_duringDD_part3.smrx'
mouse_description = 'Nms'
mouse_id = 16

#Update more rarely
experiment_id = experimental_path[:6]
file_id = file[:6]
dpa_z=0.9
target_fs=100
epoch_seconds = 2
smoothing_window = 21
random_epoch_size = 20000
lda_components = 3
dpa_k_max=201
knn_n_neighbors = 201
quantile_norm = 0.01
eps = 2
min_samples = 100


#No need to update normally
lda_figure_title_no_labels = 'LDA no labels m{}-{}_{}'.format(mouse_id, experiment_id, file_id)
lda_figure_title_dpc_labels = 'LDA DPC labels m{}-{}_{}'.format(mouse_id, experiment_id, file_id)
lda_figure_title_state_labels = 'LDA state labels m{}-{}_{}'.format(mouse_id, experiment_id, file_id)
lda_figure_title_outliers_labels = 'LDA state labels with outliers m{}-{}_{}'.format(mouse_id, experiment_id, file_id)

eeg_figure_title = 'EEG m{}-{}_{} '.format(mouse_id, experiment_id, file_id, )

ann_folder = 'D:/Project_mouse/Ongoing_analysis/ANN_training/'


# standard functions for plotting
plot_kwds = {'alpha': 0.25, 's': 20, 'linewidths': 0}
# Set figure resolution
dpi = 500

# base_path = os.getcwd().replace("\\", "/") + "/"
project_path = 'C:/Users/bitsik0000/PycharmProjects/oessa'
offline_data_path = project_path + '/data/'

average_states_path = offline_data_path + 'StateAverages.pkl'
average_knn_path = offline_data_path + 'knn_average.joblib'
average_lda_path = offline_data_path + 'lda_average.joblib' #use the version with the underscore for older versions of python

state_df_filename = base_path + experimental_path + 'states_{}_{}_{}_m{}.pkl'.format(experiment_id, file_id, mouse_description, mouse_id)
lda_filename = base_path + experimental_path + 'lda_{}_{}_{}_m{}.joblib'.format(experiment_id, file_id, mouse_description, mouse_id)
knn_filename = base_path + experimental_path + 'knn_{}_{}_{}_m{}.joblib'.format(experiment_id, file_id, mouse_description, mouse_id)