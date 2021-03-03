#!/usr/bin/env python
# coding: utf-8

# # Exercice 1 : Classification challenge

# In[11]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import learning_curve, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# import warnings
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


data_train = pd.read_csv('Spotify_train_dataset.csv')
data_test = pd.read_csv('Spotify_test_dataset.csv')
# warnings.filterwarnings('ignore')


# In[13]:


y_train = data_train['genre']
X_train = data_train.drop(data_train.select_dtypes('object').columns, axis = 1)
features = X_train.columns


# # Data analysis

# ## Global information

# In[14]:


print(data_train.shape)
print(data_test.shape)


# In[15]:


data_train.head(10)


# In[16]:


data_train.info()


# In[17]:


data_train.describe()


# In[8]:


print("Feature type : ")
data_train.dtypes.value_counts()


# ## Target analysis

# In[9]:


data_train['genre'].value_counts()


# ## Distribution of quantitative features

# In[10]:


for col in data_train.select_dtypes(['float', 'int']):
    sns.displot(data_train[col])


# In[11]:


sns.pairplot(data_train.iloc[np.random.randint(0, data_train.shape[0], size = 150), :], hue = 'genre', 
             corner = True)


# ## Qualitative features

# In[12]:


data_train.select_dtypes('object')


# ## Outliers

# In[13]:


detector = IsolationForest(contamination = 'auto')
detector.fit(X_train)
data_train[detector.predict(X_train) == -1]['genre'].value_counts()


# ## Correlation

# In[14]:


plt.figure(figsize = (12, 8))
plt.imshow(data_train.corr(), cmap = 'Blues')
plt.colorbar()


# ## Visualisation

# In[16]:


X_embedded = TSNE(n_components = 2, n_jobs = -1).fit_transform(X_train)


# In[17]:


plt.figure(figsize = (12, 8))
# Don't run the cell before encoding y_train
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c = y_train)


# In[18]:


pca = PCA(n_components = 2)
X_embedded_ = pca.fit_transform(X_train)
print(pca.explained_variance_ratio_)


# In[19]:


plt.figure(figsize = (12, 8))
# Don't run the cell before encoding y_train
plt.scatter(X_embedded_[:, 0], X_embedded_[:, 1], c = y_train)


# ### Observations
# 
# - **Target variable** : 'genre'
# - **Shape** : (31728, 20) for train set, (10577, 19) for test set
# - **Missing values** : Just for 'song_name', which is not an important feature for the classification
# 
# - **Target analysis** : quite unbalanced classes (except for 'pop' : only 336 samples)
# - **Quantitative features** : between 0 and 1 for the most part (already pre-processed)
# - **Qualitative features** : uninteresting (except 'genre' the target)

# # Preprocessing and Feature selection

# ## Encodage

# In[18]:


encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)


# ## Normalisation

# In[19]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_train_c = X_train


# ## Idea - Implementation - Evaluation cycle

# In[20]:


# A function to evaluate the model

def evaluation(model_name, model):
    y_pred = cross_val_predict(model, X_train, y_train, cv = 5, n_jobs = -1)
    plt.figure(figsize=(12,8))
    cm = confusion_matrix(y_train, y_pred)
    cm_sum = cm.sum(axis=1)
    c_matrix = cm/cm_sum.reshape((cm_sum.shape[0], 1))
    print(sns.heatmap(c_matrix, annot = True, fmt = '.2f', cmap = plt.cm.Blues))
    print("\n\nClassification report :\n\n", classification_report(y_train, y_pred))
    print('f1 score with cross_val :', cross_val_score(model, X_train, y_train,
                                                        cv = 10, scoring = 'f1_micro', 
                                                        n_jobs = -1).mean())
    
    N, train_score, val_score = learning_curve(model, X_train, y_train, cv = 10, 
                                               scoring = 'f1_micro',
                                               train_sizes = np.linspace(0.1, 1, 10), random_state = 0, n_jobs = -1)
    plt.figure(figsize = (12, 8))
    plt.plot(N, train_score.mean(axis=1), label = 'Train score')
    plt.plot(N, val_score.mean(axis=1), label = 'Val score')
    plt.xlabel('Number of data')
    plt.ylabel('F1 score')
    plt.title(f'Learning curve for {model_name}')
    plt.legend()


# ### First basic model to test ideas

# In[22]:


first_model = DecisionTreeClassifier(random_state=0)


# In[23]:


evaluation('Decision Tree', first_model)


# ### From this first model, we notice that :
# 
# - Difficulties to classify correctly, especially for classes 0 to 7 (very low score for class 3)
# - Overfitting

# ## Feature selection

# In[24]:


first_model.fit(X_train, y_train)
feature_importances = pd.DataFrame(first_model.feature_importances_, index = features)
feature_importances.plot.bar(figsize=(12,8))


# ### There are several useless variables for the model  : 
# 
# - Key, mode, time_signature
# 
# - Let's drop some features and see if it resolves the overfitting problem

# ### Correlation features / target

# In[25]:


plt.figure(figsize = (12, 8))
train = np.concatenate((X_train, y_train.reshape((y_train.shape[0], 1))), axis = 1)
train = pd.DataFrame(train)
plt.imshow(train.corr(), cmap = 'Blues')
plt.colorbar()


# ### Attempts

# In[26]:


# Try with 10 features
selector = SelectKBest(f_classif, k = 10)
X_train = selector.fit_transform(X_train, y_train)
print(selector.get_support())


# In[27]:


evaluation('Decision Tree', first_model)


# In[28]:


# Try with 5 features
X_train = X_train_c
selector = SelectKBest(f_classif, k = 5)
X_train = selector.fit_transform(X_train, y_train)
print(selector.get_support())


# In[29]:


evaluation('Decision Tree', first_model)


# In[30]:


# Try a polynomial extension
X_train = X_train_c
poly = PolynomialFeatures(degree = 2)
X_train = poly.fit_transform(X_train)
selector = SelectKBest(f_classif, k = 25)
X_train = selector.fit_transform(X_train, y_train)


# In[31]:


evaluation('Decision Tree', first_model)


# ### Conclusion :
# 
# - Feature selection and polynomial extension do not improve the performance of our basic model (a decision tree)
# - The model seems to still be in overfitting -> try a random forest
# - We are going to test again some feature selection on different models

# # Model selection 

# In[21]:


SVM = SVC()
RandomForest = RandomForestClassifier(random_state = 0)
QDA = QuadraticDiscriminantAnalysis()
AdaBoost = AdaBoostClassifier()
KNN = KNeighborsClassifier()
LogisticReg = LogisticRegression()
GNB = GaussianNB()
MLP = MLPClassifier()
models = {'SVM' : SVM, 'RandomForest' : RandomForest, 'QDA' : QDA, 'AdaBoost' : AdaBoost, 'KNN' : KNN,
          'LogisticReg' : LogisticReg, 'GNB' : GNB, 'MLP' : MLP}


# ## Try with polynomial extension (degree 2) and selection of the 10 best features

# In[33]:


X_train = X_train_c
poly = PolynomialFeatures(degree = 2)
X_train = poly.fit_transform(X_train)
selector = SelectKBest(f_classif, k = 10)
X_train = selector.fit_transform(X_train, y_train)


# In[34]:


for name, model in models.items():
    print(name, ' : ', cross_val_score(model, X_train, y_train, cv = 5, scoring = 'f1_micro', n_jobs = -1).mean())


# ## Try with polynomial extension and selection of the 25 best features

# In[35]:


X_train = X_train_c
poly = PolynomialFeatures(degree = 2)
X_train = poly.fit_transform(X_train)
selector = SelectKBest(f_classif, k = 25)
X_train = selector.fit_transform(X_train, y_train)


# In[36]:


for name, model in models.items():
    print(name, ' : ', cross_val_score(model, X_train, y_train, cv = 5, scoring = 'f1_micro', n_jobs = -1).mean())


# ## Try with our basic features

# In[22]:


X_train = X_train_c 
for name, model in models.items():
    print(name, '\n')
    evaluation(name, model)


# ### Observations :
# 
# - SVM, RandomForest and MLP are the most promising models
# - No feature selection or polynomial extension needed : the models perform better with the basic variables from the dataset (except maybe for the SVM)

# # Model optimization

# ## SVM

# In[38]:


X_train = X_train_c 
optimized_SVM = make_pipeline(PolynomialFeatures(2), SelectKBest(k = 25), SVC())


# In[39]:


params = {'svc__C' : [0.1, 10, 100], 'svc__gamma': [0.001, 0.0001]}
svm_grid = GridSearchCV(optimized_SVM, params, scoring = 'f1_micro', cv = 5, n_jobs = -1)
svm_grid.fit(X_train, y_train)


# In[40]:


print(svm_grid.best_estimator_)
print(svm_grid.best_score_)


# ### Observations :
# - The SVM model is slightly better with optimization but not significantly
# - Not overfitting which is a good point, but underfitting
# - Class 3 is a real problem to classify

# ## Random Forest

# In[41]:


X_train = X_train_c 
RandomForest = RandomForestClassifier(random_state = 0, n_jobs = -1)
params = {'n_estimators' : [10, 100, 1000], 'criterion' : ['gini', 'entropy'], 'max_depth' : [None, 10, 20, 30]}
tree_grid = GridSearchCV(RandomForest, params, scoring = 'f1_micro', cv = 5, n_jobs = -1)
tree_grid.fit(X_train, y_train)


# In[42]:


print(tree_grid.best_estimator_)
print(tree_grid.best_score_)


# In[43]:


# Evaluate precisely this optimized model
RandomForest = RandomForestClassifier(n_estimators = 1000, max_depth = 10, random_state = 0, n_jobs = -1)
evaluation('Random Forest', RandomForest)


# ### Other attempts

# In[44]:


RandomForest = RandomForestClassifier(n_estimators = 1500, max_depth = 14, random_state = 0, n_jobs = -1)
cross_val_score(RandomForest, X_train, y_train, cv = 5, scoring = 'f1_micro', n_jobs = -1).mean()


# In[45]:


RandomForest = RandomForestClassifier(n_estimators = 2000, max_depth = 14, random_state = 0, n_jobs = -1)
cross_val_score(RandomForest, X_train, y_train, cv = 5, scoring = 'f1_micro', n_jobs = -1).mean()


# ### Observations : 
# 
# - RandomForest gives better results than SVM
# - The model is still overfitting
# - Class 3 is a real problem to classify

# ## Logistic Regression

# In[46]:


# After several attempts using GridSearchCV, we end up with this model
X_train = X_train_c 
LogRegression = LogisticRegression(penalty ='l2', C = 2000, max_iter = 10000, n_jobs = -1)
log_regression = make_pipeline(PolynomialFeatures(2), SelectKBest(k = 70), LogRegression)
cross_val_score(LogRegression, X_train, y_train, cv = 5, scoring = 'f1_micro', n_jobs = -1).mean()


# In[47]:


evaluation('Logistic Regression', LogRegression)


# ## MLP

# In[59]:


parameters = {
    'hidden_layer_sizes': [(100,)*i for i in range(10)]+[(40,)*i for i in range(10)], 
    'activation': ['logistic', 'relu'], 
    'alpha': [10**-i for i in range(10)]
    }

clf = GridSearchCV(MLPClassifier(), parameters, scoring='f1_micro', n_jobs=-1)

clf.fit(X_train, y_train)


# In[62]:


clf.best_score_, clf.best_params_


# ## Ensemble method (stacking)

# In[49]:


X_train = X_train_c 
LogRegression = LogisticRegression(penalty ='l2', C = 2000, max_iter = 10000, n_jobs = -1)
log_regression = make_pipeline(PolynomialFeatures(2), SelectKBest(k = 70), LogRegression)

estimators = [
    ('RandomForest', RandomForestClassifier(n_estimators = 1500,  max_depth = 14, random_state = 0, n_jobs = -1)),
    ('QDA', QuadraticDiscriminantAnalysis()),
    ('LogisticRegression', log_regression)
]

stacking_model = StackingClassifier(estimators = estimators, final_estimator = MLPClassifier())
scores = cross_val_score(stacking_model, X_train, y_train, cv = 5, scoring = 'f1_micro', n_jobs = -1)
print(scores)
print(scores.mean())
print(scores.std())


# In[9]:


X_train = X_train_c
LogRegression = LogisticRegression(penalty ='l2', C = 2000, max_iter = 10000, n_jobs = -1)
log_regression = make_pipeline(PolynomialFeatures(2), SelectKBest(k = 70), LogRegression)
MLP = MLPClassifier(activation = 'logistic', alpha = 1e-06, hidden_layer_sizes = (100, 100, 100))

estimators = [
    ('RandomForest', RandomForestClassifier(n_estimators = 1500,  max_depth = 14, random_state = 0, n_jobs = -1)),
    ('QDA', QuadraticDiscriminantAnalysis()),
    ('LogisticRegression', log_regression),
    ('GNB', GaussianNB()),
    ('MLP', MLP)
]

stacking_model_2 = StackingClassifier(estimators = estimators, final_estimator = MLP)


scores = cross_val_score(stacking_model_2, X_train, y_train, cv = 5, scoring = 'f1_micro', n_jobs = -1)
print(scores)
print(scores.mean())
print(scores.std())


# # Final model

# In[51]:


final_model = stacking_model_2


# In[52]:


evaluation('Stacking Model', final_model)


# # Prediction for the test set

# In[53]:


final_model.fit(X_train, y_train)


# In[55]:


X_test = data_test.drop(data_test.select_dtypes('object').columns, axis = 1)
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)


# In[56]:


predictions = final_model.predict(X_test)
predictions = encoder.inverse_transform(predictions)


# In[57]:


predictions_csv = pd.DataFrame(predictions, columns = ['genre']).to_csv('./final_predictions.csv', header=True, index=False)


# In[ ]:




