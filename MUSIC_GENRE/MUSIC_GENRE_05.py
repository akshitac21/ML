#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as skm
import sklearn.model_selection as skms
import sklearn.preprocessing as skp
import sklearn.ensemble as ske


# In[2]:


music_data = pd.read_csv('/home/akshita/python-ws/Data/features_30_sec.csv')
print("Dataset has", music_data.shape)
music_data.head()


# In[3]:


# Computing the  Correlation Matrix using only the features that have 'mean' in their name.
mean_cols = [col for col in music_data.columns if 'mean' in col]
corr = music_data[mean_cols].corr()

#the matplotlib figure
f, ax = plt.subplots(figsize=(16, 11));
sns.heatmap(corr, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.title('Correlation Heatmap (MEAN variables)', fontsize = 20)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10);
plt.savefig("Corr_Heatmap.png")


# In[4]:


# A box plot to show the distribution of music pieces in different genres in different tempo/beats per minute values.
x = music_data[["label", "tempo"]]

fig, ax = plt.subplots(figsize=(16, 8));
sns.boxplot(x = "label", y = "tempo", data = x);

plt.title('Beats Per Minute Boxplot for Genres', fontsize = 20)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 10);
plt.xlabel("Genre", fontsize = 15)
plt.ylabel("BPM", fontsize = 15)
plt.savefig("BPM_Boxplot.png")


# In[5]:


#Checking for null values in the data. 
print("Columns with null values are",list(music_data.columns[music_data.isnull().any()]))
#Since there are no null values in any of the columns, there is no need to create dummy values.


# In[6]:


# map labels to index
label_index = dict()
index_label = dict()
for i, x in enumerate(music_data.label.unique()):
    label_index[x] = i
    index_label[i] = x
print(label_index)
print(index_label)


# In[7]:


# update labels in df to index
music_data.label = [label_index[l] for l in music_data.label]


# In[8]:


#Splitting the data into training testing and validation sets
data_shuffle = music_data.sample(frac=1, random_state=12).reset_index(drop=True) #shuffling the data
# remove columns that have no impact on the prediction
data_shuffle.drop(['filename', 'length'], axis=1, inplace=True)
data_y = data_shuffle.pop('label')
data_X = data_shuffle


# split original dataset into train and test
X_train, data_test_valid_X, y_train, data_test_valid_y = skms.train_test_split(data_X, data_y, train_size=0.7, random_state=12, stratify=data_y)
#split test dataset into validation and test
X_validation, X_test, y_validation, y_test = skms.train_test_split(data_test_valid_X, data_test_valid_y, train_size=0.66, random_state=12, stratify=data_test_valid_y)
#stratify - all the sets will have an equal proportion of genres

#Validating the data splits
print(f"Train set has {X_train.shape[0]} records out of {len(data_shuffle)} which is {round(X_train.shape[0]/len(data_shuffle)*100)}% of the data.")
print(f"Validation set has {X_validation.shape[0]} records out of {len(data_shuffle)} which is {round(X_validation.shape[0]/len(data_shuffle)*100)}% of the data.")
print(f"Test set has {X_test.shape[0]} records out of {len(data_shuffle)} which is {round(X_test.shape[0]/len(data_shuffle)*100)}% of the data.")


# In[9]:


#scaling the features. This will ensure that features with larger values do not affect the prediction
#disproportionately compared to the smaller values
scaler = skp.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_validation = pd.DataFrame(scaler.transform(X_validation), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)


# In[10]:


X_train[:5] # To show the changes in the values of the features.


# In[11]:


def results(model, X, validation=False):
    y_true = y_train
    if validation:
        X = X_validation[X.columns]
        y_true = y_validation
    y_prediction = model.predict(X)
    confusion_matrix = skm.confusion_matrix(y_true, y_prediction)
    cm_display = skm.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
    cm_display.plot()
    plt.show()
    print(skm.classification_report(y_true, y_prediction, digits=3))
    print(skm.precision_recall_fscore_support(y_true, y_prediction, average="macro"))
    print("=====================================================")


# In[19]:


#Feature importance using logistic regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1999)
lr.fit(X_train, y_train)
results(lr, X_train)
results(lr, X_train, validation=True)


# In[20]:


import eli5
from eli5.sklearn import PermutationImportance


# In[21]:


perm = PermutationImportance(lr, random_state=12).fit(X_validation, y_validation, n_iter=10)
print("Feature Importances using Permutation Importance from the eli5 library")
eli5.show_weights(perm, feature_names = X_train.columns.tolist())


# In[22]:


# plot the permutation importances
perm_indices = np.argsort(perm.feature_importances_)[::-1]
perm_features = [X_validation.columns.tolist()[xx] for xx in perm_indices]
plt.figure(figsize=(14, 14))
plt.title("Logistic Regression feature importance via permutation importance")
plt.barh(range(X_validation.shape[1]), perm.feature_importances_[perm_indices])
plt.yticks(range(X_validation.shape[1]), perm_features)
plt.ylim([X_validation.shape[1], -1])
plt.savefig("perm_importance.png")
plt.show()


# In[25]:


# build model using perm selected top 30 features
lr_mif = LogisticRegression(max_iter=1999)
X_train_mif = X_train[perm_features[:30]]
lr_mif.fit(X_train_mif,y_train)
results(lr_mif, X_train_mif)
results(lr_mif, X_train_mif, validation=True)


# In[26]:


# Tuning hyperparameters of the logistic regression model using grid_search. 
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty =['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]

param_grid = {'solver': solvers, 'penalty':penalty, 'C': c_values}
cv = skms.RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
lr_grid_search = skms.GridSearchCV(lr_mif, param_grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
lr_grid_search.fit(X_train_mif, y_train)
print(lr_grid_search.best_params_)
print(lr_grid_search.best_estimator_)


# In[27]:


results(lr_grid_search.best_estimator_, X_train_mif)
results(lr_grid_search.best_estimator_, X_train_mif, validation=True)


# In[28]:


# Fitting the dataset with the 30 most important features to a random forest model
rfc = ske.RandomForestClassifier(random_state=12, n_jobs=-1)
rfc.fit(X_train_mif, y_train)
results(rfc, X_train_mif)
results(rfc, X_train_mif, validation=True)


# In[29]:


# Tuning hyperparameters of the random forest model using grid_search.
n_estimators = [100, 1000]
max_features = ['sqrt', 'log2']
param_grid_rf = {'n_estimators': n_estimators, 'max_features': max_features}
cv = skms.RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
rf_grid_search = skms.GridSearchCV(estimator=rfc, param_grid=param_grid_rf, cv=cv, scoring='accuracy',error_score=0)
rf_grid_search.fit(X_train_mif, y_train)
print(rf_grid_search.best_params_)
print(rf_grid_search.best_estimator_)


# In[30]:


results(rf_grid_search.best_estimator_, X_train_mif)
results(rf_grid_search.best_estimator_, X_train_mif, validation=True)


# In[31]:


# Modifying the results function to return a confusion matrix for the test set
def test_results(model, X, validation=False):
    y_true = y_train
    if validation:
        X = X_test[X.columns]
        y_true = y_test
    y_prediction = model.predict(X)
    confusion_matrix = skm.confusion_matrix(y_true, y_prediction)
    cm_display = skm.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
    cm_display.plot()
    plt.show()
    print(skm.classification_report(y_true, y_prediction, digits=3))
    print(skm.precision_recall_fscore_support(y_true, y_prediction, average="macro"))
    print("=====================================================")



# In[32]:


# Calculating the accuracy, precision, recall, f1-score, and sensitivity of the random forest model.
test_results(rf_grid_search.best_estimator_, X_train_mif, validation=True)



# In[34]:


import librosa, librosa.display
import IPython.display as ipd

plt.rcParams['figure.figsize'] = (10, 3)

# from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist


# In[35]:


features = music_data.iloc[:,2:-1]
features_scaled=skp.scale(features)

features.describe().T


# In[38]:


distances = cdist(features_scaled, features_scaled, 'cosine') #cosine distance

dist_df = pd.DataFrame(distances)
dist_df = dist_df.set_index(music_data.filename)
dist_df.columns = music_data.filename
dist_df


# In[39]:


def songs_similarity(song_name, features, metric='cosine'):
    distances = cdist(features, features, metric=metric)
    dist_df = pd.DataFrame(distances)
    dist_df = dist_df.set_index(music_data.filename)
    dist_df.columns = music_data.filename
    series = dist_df[song_name].sort_values(ascending = True)
    series = series.drop(song_name)
    return series


# In[40]:


audio_path = '/home/akshita/python-ws/Data/genres_original/'
song_name = 'classical.00077.wav'
ipd.Audio(audio_path+song_name.split('.')[0]+'/'+song_name)



# In[41]:


sim_songs = songs_similarity(song_name, features_scaled)
sim_songs


# In[42]:


print('- 3 most similar songs: -')
for i in range(3):
    sim_song = sim_songs.index[i]
    print(sim_song)
    ipd.display(ipd.Audio(audio_path+sim_song.split('.')[0]+'/'+ sim_song))


# In[43]:


print('- 3 most different songs: -')
for i in range(1,4):
    sim_song = sim_songs.index[-i]
    print(sim_song)
    ipd.display(ipd.Audio(audio_path+sim_song.split('.')[0]+'/'+ sim_song))


# In[44]:


song_name = 'pop.00030.wav'
ipd.Audio(audio_path+song_name.split('.')[0]+'/'+song_name)


# In[48]:


sim_songs = songs_similarity(song_name, features_scaled)
sim_songs


# In[46]:


print('- 3 most similar songs: -')
for i in range(3):
    sim_song = sim_songs.index[i]
    print(sim_song)
    ipd.display(ipd.Audio(audio_path+sim_song.split('.')[0]+'/'+ sim_song))


# In[47]:


print('- 3 most different songs: -')
for i in range(1,4):
    sim_song = sim_songs.index[-i]
    print(sim_song)
    ipd.display(ipd.Audio(audio_path+sim_song.split('.')[0]+'/'+ sim_song))


# In[ ]:




