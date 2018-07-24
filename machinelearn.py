import os
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats, linalg
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.covariance import GraphLasso
from sklearn.linear_model import Ridge, BayesianRidge, Lasso, MultiTaskLasso, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


#split data in to very preterm and slightly preterm:
def splitdataga(labels,features):
    labels = np.array(labels)
    labels_vpterm = labels[np.where(labels<28)]
    features_vpterm = features[np.where(labels<28)]
    labels_spterm = labels[np.where(labels>30)]
    features_spterm = features[np.where(labels>30)]
    return labels_vpterm, features_vpterm, labels_spterm, features_spterm


#get the PCA of the data:
def getPCA(arr):
    pca = PCA(n_components=1)
    pca.fit(arr)
    pca_data = pca.transform(arr)
    #'''
    #get the values of explained variance and the principal components:
    per_var=np.round(pca.explained_variance_ratio_* 100, decimals = 1)
    labels = [str(x) for x in range(1, len(per_var)+1)]
    #plot explained variance:
    plt.figure(figsize=(20,10))
    plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained variance')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot')
    plt.show()
    #'''
    return pca_data

def getdata(dataname):
    #get the database for machine learning:
    directory = "C:\\Users\\surya\\OneDrive - King's College London\\KURF"
    file_datas = os.path.join(directory, dataname)
    database = pd.read_csv(file_datas)
    labels_ga = database['Gestation age'][:].tolist()
    #'''
    features_pma = database['Age at scan'][:].as_matrix()-np.mean(database['Age at scan'][:].as_matrix())
    features = database.iloc[:, 5:].as_matrix()
    features = np.concatenate((features_pma.reshape(features_pma.shape[0], 1), features), axis=1)
    #'''
    labels_ga=np.array(labels_ga)
    labels_ga = labels_ga[np.where(np.abs(features_pma)<6)]
    features = features[np.where(np.abs(features_pma)<6)]
    #'''
    #'''
    print('Debug')
    return labels_ga, features


#Crossvalidation for each model type: 
def crossval(labels,features,algorithm):
    features = np.nan_to_num(features)
    #features = getPCA(np.nan_to_num(features))
    #plot scatter of the first column of the features array (assuming the pca has been done):
    #'''
    plt.figure(figsize=(20,10))
    plt.scatter(x=labels,y=features[:,0])
    plt.xlabel('Gestational, age')
    plt.ylabel('PCA score')
    plt.title('Gestational age vs principal component one')
    plt.show()
    #'''
    alpha_num = (0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 5.0, 10.0, 50.0, 100.0)
    DATA_train, DATA_test, LABELS_train, LABELS_test = train_test_split(
        features, labels, test_size=0.1, random_state=42)
    
    if algorithm == 1:
        model = MultiTaskLasso() 
        param_grid = {'alpha': np.random.uniform(0.1,100,100),
              'fit_intercept': [True, False],
              'normalize': [True, False],
              'max_iter': np.linspace(100,10000,num=50,dtype=int),
              'tol': np.linspace(0.000001,0.001,num=50)}
        clf = RandomizedSearchCV(model, param_distributions=param_grid,
                                 cv=10, scoring='r2', random_state=42, n_iter=100)
        clf.fit(DATA_train, LABELS_train)
        paramopti = clf.best_params_
        score = clf.best_score_
        score_grid = clf.cv_results_   
        return paramopti, score, score_grid  
    elif algorithm == 2:
        model = ElasticNet()
        param_grid = {'alpha': np.random.uniform(0.1,100,1000),
              'l1_ratio': np.random.uniform(0,1,1000)}
        clf = RandomizedSearchCV(model, param_distributions=param_grid,
                                 cv=10, scoring='r2', random_state=42, n_iter=100)
        clf.fit(DATA_train, LABELS_train)
        paramopti = clf.best_params_
        score = clf.best_score_
        score_grid = clf.cv_results_   
        return paramopti, score, score_grid 
    elif algorithm == 3:
        model = BayesianRidge()
        param_grid = {'alpha_1': np.random.uniform(0.000001,10,1000),
              'alpha_2': np.random.uniform(0.000001,10,1000),
              'lambda_1': np.random.uniform(0.000001,10,1000),
              'lambda_2': np.random.uniform(0.000001,10,1000),
              'fit_intercept': [True, False],
              'normalize': [True, False],
              'max_iter': np.linspace(100,10000,num=50,dtype=int),
              'tol': np.linspace(0.000001,0.001,num=50)}
        clf = RandomizedSearchCV(model, param_distributions=param_grid,
                                 cv=10, scoring='r2', random_state=42, n_iter=100)
        clf.fit(DATA_train, LABELS_train)
        paramopti = clf.best_params_
        score = clf.best_score_
        score_grid = clf.cv_results_   
        return paramopti, score, score_grid     
    elif algorithm == 4:
        model = Lasso()
        param_grid = {'alpha': np.random.uniform(0.1,100,100),
              'fit_intercept': [True, False],
              'normalize': [True, False],
              'max_iter': np.linspace(100,10000,num=50,dtype=int),
              'tol': np.linspace(0.000001,0.001,num=50)}
        clf = RandomizedSearchCV(model, param_distributions=param_grid,
                                 cv=10, scoring='r2', random_state=42, n_iter=100)
        clf.fit(DATA_train, LABELS_train)
        paramopti = clf.best_params_
        score = clf.best_score_
        score_grid = clf.cv_results_   
        return paramopti, score, score_grid     
    elif algorithm == 5:
        model = SVR() 
        param_grid = {'C': np.random.uniform(0.1,10000,100),
              'kernel': ['rbf','poly','sigmoid'],
              'normalize': [True, False],
              'max_iter': np.arange(100,10000,100,dtype=int),
              'tol': np.arange(0.000001,0.001,0.000001)}
        clf = RandomizedSearchCV(model, param_distributions=param_grid,
                                 cv=10, scoring='r2', random_state=42, n_iter=100)
        clf.fit(DATA_train, LABELS_train)
        paramopti = clf.best_params_
        score = clf.best_score_
        score_grid = clf.cv_results_   
        return paramopti, score, score_grid  
        
    elif algorithm == 6:
        model = RandomForestRegressor()
        model.fit(DATA_train, LABELS_train)
        depth_range = np.arange(2,model.n_features_, np.int(0.1*model.n_features_)+1)
        estimator_range = np.arange(20, 2020, 80)
        feature_range = np.linspace(1,model.n_features_,0.4*model.n_features_,dtype=int)
        param_grid = dict(
            max_depth=depth_range, n_estimators=estimator_range, max_features=feature_range)
        clf = RandomizedSearchCV(model, param_distributions=param_grid,
                                 cv=10, scoring='r2', random_state=42, n_iter=20)
        clf.fit(DATA_train, LABELS_train)
        paramopti = clf.best_params_
        score = clf.best_score_
        score_grid = clf.cv_results_   
        return paramopti, score, score_grid      
    elif algorithm == 7:
        model = Ridge()
        param_grid = {'alpha': np.random.uniform(0.1,100,1000),}
        clf = GridSearchCV(model, param_grid=param_grid,
                                 cv=10, scoring='r2', return_train_score=True)
        clf.fit(DATA_train, LABELS_train)
        paramopti = clf.best_params_
        score = clf.best_score_
        score_grid = clf.cv_results_        
        return paramopti, score, score_grid
    elif algorithm == 8:
        clf = LinearRegression()
        clf.fit(DATA_train, LABELS_train)
        clf.predict(DATA_test)
        return clf.score(DATA_test,LABELS_test)
    elif algorithm == 9:
        model = SVR()
        param_grid = {'C': np.random.uniform(0.1,2000,100),
              'kernel': ['linear','rbf','poly','sigmoid'],
              'max_iter': np.arange(100,2000,100,dtype=int),
              'tol': np.linspace(0.000001,0.001,num=50)}
        clf = RandomizedSearchCV(model, param_distributions=param_grid,
                                 cv=10, scoring='r2', random_state=42, n_iter=100)
        clf.fit(DATA_train, LABELS_train)
        paramopti = clf.best_params_
        score = clf.best_score_
        score_grid = clf.cv_results_        
        return paramopti, score, score_grid
    else:
        print('Algorithms not recognised')
    
    
        
'''
labels_ga, features_parcorr = getdata('Partial correlation.csv')
labels_ga, features_corr = getdata('Correlation.csv')
labels_ga, features_cov = getdata('Covariance.csv')
#p_corr_paramopti_rf,p_corr_score_rf,p_corr_grid_rf = crossval(labels_ga, features_parcorr, 6)
p_corr_paramopti_en,p_corr_score_en,p_corr_grid_en = crossval(labels_ga, features_parcorr, 2)
p_corr_paramopti_ri,p_corr_score_ri,p_corr_grid_ri = crossval(labels_ga, features_parcorr, 7)
p_corr_paramopti_las,p_corr_score_las,p_corr_grid_las = crossval(labels_ga, features_parcorr, 4)
#p_corr_paramopti_bay_ri,p_corr_score_bay_ri,p_corr_grid_bay_ri = crossval(labels_ga, features_parcorr, 3)
#p_corr_paramopti_mult_las,p_corr_score_mult_las,p_corr_grid_mult_las = crossval(labels_ga, features_parcorr, 1)
p_corr_score_lin = crossval(labels_ga, features_parcorr, 8)
#corr_paramopti_rf,corr_score_rf,corr_grid_rf = crossval(labels_ga, features_corr, 6)
corr_paramopti_en,corr_score_en,corr_grid_en = crossval(labels_ga, features_corr, 2)
corr_paramopti_ri,corr_score_ri,corr_grid_ri = crossval(labels_ga, features_corr, 7)
corr_paramopti_las,corr_score_las,corr_grid_las = crossval(labels_ga, features_corr, 4)
#corr_paramopti_bay_ri,corr_score_bay_ri,corr_grid_bay_ri = crossval(labels_ga, features_corr, 3)
#corr_paramopti_mult_las,corr_score_mult_las,corr_grid_mult_las = crossval(labels_ga, features_corr, 1)
corr_score_lin = crossval(labels_ga, features_corr, 8)
#cov_paramopti_rf,cov_score_rf,cov_grid_rf = crossval(labels_ga, features_cov, 6)
cov_paramopti_en,cov_score_en,cov_grid_en = crossval(labels_ga, features_cov, 2)
cov_paramopti_ri,cov_score_ri,cov_grid_ri = crossval(labels_ga, features_cov, 7)
cov_paramopti_las,cov_score_las,cov_grid_las = crossval(labels_ga, features_cov, 4)
#cov_paramopti_bay_ri,cov_score_bay_ri,cov_grid_bay_ri = crossval(labels_ga, features_cov, 3)
#cov_paramopti_mult_las,cov_score_mult_las,cov_grid_mult_las = crossval(labels_ga, features_cov, 1)
cov_score_lin = crossval(labels_ga, features_cov, 8)
print('Debug')
#'''

labels_ga, features = getdata('Volumes.csv')
#'''
#paramopti_rf,score_rf,grid_rf = crossval(labels_ga, features, 6)
paramopti_en,score_en,p_corr_grid_en = crossval(labels_ga, features, 2)
paramopti_ri,score_ri,grid_ri = crossval(labels_ga, features, 7)
paramopti_las,score_las,grid_las = crossval(labels_ga, features, 4)
#'''
paramopti_svm,score_svm,grid_svm = crossval(labels_ga, features, 9)
score_lin = crossval(labels_ga, features, 8)


'''
labels_ga, features = getdata('Volumes.csv')
labels_ga_vp, features_vp, labels_ga_sp, features_sp = splitdataga(labels_ga, features)
#paramopti_rf_vp,score_rf_vp,grid_rf_vp = crossval(labels_ga_vp, features_vp, 6)
paramopti_en_vp,score_en_vp,p_corr_grid_en_vp = crossval(labels_ga_vp, features_vp, 2)
paramopti_ri_vp,score_ri_vp,grid_ri_vp = crossval(labels_ga_vp, features_vp, 7)
paramopti_las_vp,score_las_vp,grid_las_vp = crossval(labels_ga_vp, features_vp, 4)
score_lin_vp = crossval(labels_ga_vp, features_vp, 8)
#paramopti_rf_sp,score_rf_sp,grid_rf_sp = crossval(labels_ga_vp, features_vp, 6)
paramopti_en_sp,score_en_sp,p_corr_grid_en_sp = crossval(labels_ga_sp, features_sp, 2)
paramopti_ri_sp,score_ri_sp,grid_ri_sp = crossval(labels_ga_sp, features_sp, 7)
paramopti_las_sp,score_las_sp,grid_las_sp = crossval(labels_ga_sp, features_sp, 4)
score_lin_sp = crossval(labels_ga_sp, features_sp, 8)
#'''