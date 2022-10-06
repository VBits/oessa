import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.rc('lines', linewidth=0.5)
import pandas as pd
# import sys
# sys.path.append('C:/Users/bitsik0000/PycharmProjects/ClosedLoopEEG/OfflineAnalysis')
from Utilities.Mouse import Mouse
import Config as OfflineConfig
from Utilities.GeneralUtils import query_option, expand_epochs
from Utilities.Transformations import train_lda,lda_transform_df
import joblib
import Utilities.ANN as ANN

def process_EEG_data(description, mouse_id):
    m = Mouse(description, mouse_id)

    # Create directory to save figures
    m.gen_folder(OfflineConfig.base_path, OfflineConfig.experimental_path)
    load_data = query_option("Pick option: \n1) Preprocess raw data \n2) Load previously stored pkl files?",valid_options=[1,2])
    if load_data==1:
        print('Processing EEG data and storing files: _{}_{}_{}_m{}.pkl'.format(OfflineConfig.experiment_id, OfflineConfig.file_id, m.description, m.mouse_id))
        #Load EEG data
        m.read_smrx(OfflineConfig.base_path, OfflineConfig.experimental_path, OfflineConfig.file)

        ### -------------------
        ### Downsample, perform multitaper and normalize data
        if m.EEG_fs > OfflineConfig.target_fs:
            print ('downsampling mouse {} EEG data, from {}Hz to {}Hz'.format(m.mouse_id,m.EEG_fs,OfflineConfig.target_fs))
            m.downsample_EGG(target_fs=OfflineConfig.target_fs)


        m.multitaper(resolution=OfflineConfig.epoch_seconds)
        m.smoothen_and_norm_spectrum(window_size=OfflineConfig.smoothing_window,quantile=OfflineConfig.quantile_norm)

        # Save normalized Dataframe to experimental folder
        m.Sxx_df.to_pickle(OfflineConfig.base_path + OfflineConfig.experimental_path + 'Sxx_df_{}_{}_{}_m{}.pkl'.format(OfflineConfig.experiment_id, OfflineConfig.file_id, m.description, m.mouse_id))
        m.Sxx_norm.to_pickle(OfflineConfig.base_path + OfflineConfig.experimental_path + 'Sxx_norm_{}_{}_{}_m{}.pkl'.format(OfflineConfig.experiment_id, OfflineConfig.file_id, m.description, m.mouse_id))
        m.multitaper_df.to_pickle(OfflineConfig.base_path + OfflineConfig.experimental_path + 'Multitaper_df_{}_{}_{}_m{}.pkl'.format(OfflineConfig.experiment_id, OfflineConfig.file_id, m.description, m.mouse_id))
    elif load_data ==2:
        print('Loading previously analysed file {}_{}_{}_m{}.pkl'.format(OfflineConfig.experiment_id, OfflineConfig.file_id, m.description, m.mouse_id))
        # Load previously saved Dataframe from experimental folder
        m.Sxx_df = pd.read_pickle(OfflineConfig.base_path + OfflineConfig.experimental_path + 'Sxx_df_{}_{}_{}_m{}.pkl'.format(OfflineConfig.experiment_id, OfflineConfig.file_id, m.description, m.mouse_id))
        m.Sxx_norm = pd.read_pickle(OfflineConfig.base_path + OfflineConfig.experimental_path + 'Sxx_norm_{}_{}_{}_m{}.pkl'.format(OfflineConfig.experiment_id, OfflineConfig.file_id, m.description, m.mouse_id))
        m.multitaper_df = pd.read_pickle(OfflineConfig.base_path + OfflineConfig.experimental_path + 'Multitaper_df_{}_{}_{}_m{}.pkl'.format(OfflineConfig.experiment_id, OfflineConfig.file_id, m.description, m.mouse_id))

    #Create an extended dataframe that contains the smoothed and raw epochs
    m.Sxx_ext = expand_epochs(m)
    return m


def get_LDA(m,rand_idx):
    #2. Load the average LDA or create a new LDA
    use_lda = query_option("Pick option \n1) Load average LDA \n2) Load previously trained LDA for this animal"
                           "\n3) Get provisional labels from ANN and train a new LDA",valid_options=[1,2,3])
    if use_lda==1:
        print('Using an average LDA trained on multiple animals')
        #Load a previously created LDA (works better with noisy data)
        lda = joblib.load(OfflineConfig.average_lda_path)
    elif use_lda ==2:
        print('Using an LDA trained on this animal')
        lda = joblib.load(OfflineConfig.lda_filename)
    elif use_lda == 3:
        print ('Using an ANN to get provisional labels and train a new LDA')
        ANN.label_data(m,rand_idx,reuse_weights=True)
        # Create LDA using the ANN labels (works better with noise free data)
        lda, x_train = train_lda(m.Sxx_ext, m.state_df['ann_labels'], rand_idx, components=OfflineConfig.lda_components)

    # Reduce dimensionality of data and save in a new dataframe
    m.LD_df = lda_transform_df(m.Sxx_ext, lda)
    return lda

