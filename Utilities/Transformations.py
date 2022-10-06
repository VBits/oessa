from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd


def train_lda(dataframe,labels,rand_idx,components=3):
    lda = LDA(n_components=components)
    return lda, lda.fit_transform(dataframe.loc[rand_idx], labels.loc[rand_idx])

def lda_transform_df(dataframe,lda):
    # Create dataframe for LDs
    LD = lda.transform(dataframe.values)
    if lda.n_components == 2:
        LD_df = pd.DataFrame(data=LD, columns=['LD1', 'LD2'], index=dataframe.index)
    elif lda.n_components == 3:
        LD_df = pd.DataFrame(data=LD, columns=['LD1', 'LD2', 'LD3'], index=dataframe.index)
    return LD_df

def train_lda_dpa_cores(dataframe,est,rand_idx,components=3):
    lda = LDA(n_components=components)
    #exclude halos
    dataframe_c = dataframe.loc[rand_idx][est.halos_ != -1]
    cores = est.halos_[est.halos_ != -1]
    return lda, lda.fit_transform(dataframe_c, cores)

def train_lda_dpa_labels(dataframe,est,rand_idx,components=3):
    lda = LDA(n_components=components)
    return lda, lda.fit_transform(dataframe.loc[rand_idx].values, est.labels_)
