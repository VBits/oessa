import Config as OfflineConfig
from Utilities.PlottingUtils import plot_LDA
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def get_knn_clf(m,rand_idx,est,n_neighbors=201):
    # Use Kneighbors to classify the rest of the data, use odd numbers to avoid draws
    sample_data = np.ascontiguousarray(m.LD_df.loc[rand_idx].values)
    knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_clf.fit(sample_data, est.labels_)
    return knn_clf

def propagate_knn_labels(m,rand_idx,knn_clf):
    m.knn_pred(knn_clf, m.Sxx_ext,OfflineConfig.average_states_path)
    # Plot and evaluate state assignment 3D
    plot_LDA(m,rand_idx,m.state_df['states'])