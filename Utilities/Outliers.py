"""
This code can be used to annotate outliers using DBSCAN
These labels can then be propagated using KNN to the rest of the dataset
"""
#################
from sklearn.cluster import DBSCAN

dbscan_model = DBSCAN(eps=1.8,min_samples=100).fit(m.LD_df.loc[rand_idx]) # (2, 100)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(m.LD_df['LD1'].loc[rand_idx],m.LD_df['LD2'].loc[rand_idx],m.LD_df['LD3'].loc[rand_idx],c=dbscan_model.labels_,cmap='Dark2_r',alpha=0.5, s=5)
ax.set_xlabel('LD1')
ax.set_ylabel('LD2')
ax.set_zlabel('LD3')
plt.title('LDA')
plt.savefig(m.figureFolder+'LDA and labeled outliers' + m.figure_tail, dpi=dpi)


### -----
# Predict outliers in the rest of the data
clf_outlier = KNeighborsClassifier(n_neighbors=5)
sample_data = np.ascontiguousarray(m.LD_df.loc[rand_idx].values)
clf_outlier.fit(sample_data, dbscan_model.labels_)
# predict states
m.state_df['outliers'] = clf_outlier.predict(m.LD_df)






# Annotate the state dataframe
m.state_df.loc[m.state_df['outliers']!=0,'states']= 'ambiguous'

# Validate annotation
plot_LDA(m,rand_idx,m.state_df['states'],savefigure=False)
plt.savefig(m.figureFolder+'LDA DPC labels and outliers {}_{}'.format(ExpDir[:6], File[:6]) + m.figure_tail, dpi=dpi)

#OPTIONAL save the file including epochs labeled as embiguous
m.state_df.to_pickle(BaseDir + ExpDir + 'states_{}_{}_{}_m{}.pkl'.format(ExpDir[:6], File[:6], m.genotype, m.pos))
