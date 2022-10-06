"""
This code can be used to annotate outliers using DBSCAN
These labels can then be propagated using KNN to the rest of the dataset
"""
#################
from sklearn.cluster import DBSCAN
from Utilities.PlottingUtils import plot_LDA, plot_outliers
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def outlier_detection(m,rand_idx,eps=1.8,min_samples = 100):
    outlier_model = DBSCAN(eps=eps,min_samples=min_samples).fit(m.LD_df.loc[rand_idx])

    plot_outliers(m,rand_idx,outlier_model)
    return outlier_model

def predict_all_outliers(m,rand_idx,outlier_model, ambiguous_state=0):
    ### -----
    # Predict outliers in the rest of the data
    clf_outlier = KNeighborsClassifier(n_neighbors=5)
    sample_data = np.ascontiguousarray(m.LD_df.loc[rand_idx].values)
    clf_outlier.fit(sample_data, outlier_model.labels_)
    # predict states
    m.state_df['outliers'] = clf_outlier.predict(m.LD_df.values)


    # Annotate the state dataframe
    m.state_df.loc[m.state_df['outliers']!=ambiguous_state,'states']= 'ambiguous'

    # Validate annotation
    plot_LDA(m,rand_idx,m.state_df['states'])


