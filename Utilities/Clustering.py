from Pipeline import DPA
from Utilities.PlottingUtils import plot_DPA_LDA
from Utilities.Transformations import train_lda_dpa_labels,lda_transform_df
import Config as OfflineConfig
import numpy as np


def clustering_DPA(m,rand_idx,dpa_z=OfflineConfig.dpa_z,k_max=OfflineConfig.dpa_k_max):
    """
    Density peak clustering. Find density peaks in low dimensional space, tweak Z
    :param m:
    :param rand_idx:
    :param dpa_z:
    :param k_max:
    :return:
    """
    est = DPA.DensityPeakAdvanced(Z=dpa_z,k_max=k_max)
    est.fit(m.LD_df.loc[rand_idx])

    # Plot DPA clusters on LDA
    plot_DPA_LDA(m, rand_idx, est)
    return est


def merging_spurious_labels(m,rand_idx,label_dict,est):
    #change the dictionary format
    label_dict = {vi: k for k, v in label_dict.items() for vi in v}
    #merge the labels
    est.labels_ = np.vectorize(label_dict.get)(est.labels_)
    # Plot DPA clusters on LDA
    plot_DPA_LDA(m, rand_idx, est)


def retrain_LDA(m,rand_idx,est):
    lda, x_train = train_lda_dpa_labels(m.Sxx_ext,est,rand_idx,components=OfflineConfig.lda_components)
    m.LD_df = lda_transform_df(m.Sxx_ext,lda)
    return lda