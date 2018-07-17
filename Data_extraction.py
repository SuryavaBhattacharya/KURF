
import os
import nibabel as nib
import numpy as np
import pandas as pd
#import matlab.engine
import matplotlib.pyplot as plt
from scipy import stats, linalg
from sklearn.covariance import GraphLassoCV
#import nilearn

def labelextractor(sub_id):
    excl_list=[]
    count = 0
    for i in range (0,len(sub_id.iloc[:])):
        label_file = 'EP' + str(np.int(sub_id.iloc[i])) + '_all_labels.nii.gz'            
        labelpath = os.path.join(Directory2,label_file)
        try:
            datalabel = np.unique(nib.load(labelpath).get_fdata())
        except FileNotFoundError:
            message = 'Data for ' + labelpath[labelpath.find("EP"):labelpath.find("EP")+6] + ' missing.'
            excl_list.append(labelpath[labelpath.find("EP"):labelpath.find("EP")+6])
            print(message)
        else: 
            if len(datalabel)==88:
                out = datalabel
                count = count+1
            else:
                message2 = 'Data for ' + labelpath[labelpath.find("EP"):labelpath.find("EP")+6] + ' not used.'
                print(message2)
                excl_list.append(labelpath[labelpath.find("EP"):labelpath.find("EP")+6])
    return out,excl_list,count


def triangle(n):
    return 0.5*n*(n+1)

def meantimes(data_arr,datalabel,labels):
    out = np.zeros([data_arr.shape[3],len(labels)-1])
    for i in range(0,data_arr.shape[3]):
        for j in labels[1:]:
            time_arr = data_arr[:,:,:,i]
            out[i,np.int(j)-1]=np.mean(time_arr[np.where(datalabel==j)])
    return out

def extractupper(arr):
    out = np.array([])
    for i in range(0, arr.shape[0]):
        out = np.concatenate([out, arr[i, i+1:]])
    return out
        
def partialcorr(arr):
    C = np.asarray(arr)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot(beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)

            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr
    return P_corr

def corrcov(arr,typedat):
    #eng = matlab.engine.start_matlab()
    #out = eng.partialcorr(matlab.double(arr.tolist()))
    #fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(25,15))
    #im = axes[0].imshow(out)
    #return np.array(out)
    
    #im1 = axes[0,0].imshow(P_corr)
    #fig.colorbar(im1, ax=axes[0,0])
    if typedat == 'Partial correlation':
        out = partialcorr(arr)
    elif typedat == 'GraphLassoCV covariance':
        estimator = GraphLassoCV()
        estimator.fit(arr)
        out = estimator.covariance_
    elif typedat == 'GraphLassoCV precision':
        estimator = GraphLassoCV()
        estimator.fit(arr)
        out = estimator.precision_
    elif typedat == 'Covariance':
        out = np.cov(arr.transpose())
    elif typedat == 'Correlation':
        out = np.corrcoef(arr.transpose())
    # im2 = axes[0,1].imshow(covar)
    # fig.colorbar(im2, ax=axes[0, 1])
    # im3 = axes[0,2].imshow(inverscovar)
    # fig.colorbar(im3, ax=axes[0, 2])
    # im4 = axes[1,0].imshow(covar2)
    # fig.colorbar(im4, ax=axes[1, 0])
    # im5 = axes[1,2].imshow(corr)
    # fig.colorbar(im5, ax=axes[1, 2])
    # fig.savefig('partialcorr.png', bbox_inches='tight')
    return out
    


def dataextractor(subject,labels,typedat):
    rsfmri_file = subject + '_clean_rest_to_T2.nii.gz'
    filepath = os.path.join(Directory2,rsfmri_file)
    label_file = subject + '_all_labels.nii.gz'            
    labelpath = os.path.join(Directory2,label_file)
    try:
        dataarr = nib.load(filepath).get_fdata()
        datalabel = nib.load(labelpath).get_fdata()
    except FileNotFoundError:
        message = 'Data for ' + filepath[filepath.find("EP"):filepath.find("EP")+6] + ' missing.'
        print(message)
        return
    else:
        time_avg = meantimes(dataarr,datalabel,labels)
        corr_cov_dat = corrcov(time_avg,typedat)
        return corr_cov_dat
        

#Uncomment on the work PC:
#'''
Directory1 = "/projects/perinatal/peridata/EPRIME"
Directory2 = "/home/sbh17/Documents/KURFEprime2"
#'''

#Uncomment on local PC:
'''
Directory1 = "C:\\Users\\surya\\OneDrive - King's College London\\KURF\\Data2"
Directory2 = "C:\\Users\\surya\\OneDrive - King's College London\\KURF\\Data2"
#'''

def dataframeextract(typedat):
    file_sub_csv = os.path.join(Directory2,'eprime_INFO.csv')
    subs = pd.read_csv(file_sub_csv, sep = ',')
    sub_id = subs.iloc[:,0]
    sub_id = sub_id.dropna(axis=0)
    labels,excl_list,count = labelextractor(sub_id)
    ind1 = triangle(len(labels)-2)
    feat_set = np.empty((count,np.int(ind1)))
    sub_id_final = []
    ind = 0
    gender = []
    gest_age = []
    scan_age = []
    print(typedat)
    for i in range (0,len(sub_id.iloc[:])):
        #obtain file and pathname:
        subject = 'EP' + str(np.int(sub_id.iloc[i]))
        if subject not in excl_list:
            corr_cov_mat = dataextractor(subject,labels,typedat)
            feat_vect = extractupper(corr_cov_mat)
            feat_set[ind] = feat_vect
            sub_id_final.append(subject)
            gender.append(subs.iloc[i, 5])
            gest_age.append(subs.iloc[i, 3])
            scan_age.append(subs.iloc[i, 4])
            message = subject + ' complete.'
            ind = ind + 1
            print(message)
        else:
            continue

        
    Attributes = pd.DataFrame({'Subject ID': sub_id_final,
                            'Gender': gender,
                            'Gestation age': gest_age,
                            'Age at scan': scan_age})
    labelsdata = np.arange(0,ind1)
    labelsdata = list(str(i) for i in labelsdata)
    datafromsubs = pd.DataFrame(data=feat_set,columns=labelsdata)
    finaldataf = pd.concat([Attributes, datafromsubs], axis=1)
    index = list(str(i) for i in range(0, len(sub_id_final)))
    finaldataf = finaldataf.set_index([index, 'Subject ID'])
    fname = typedat + '.csv'
    finaldataf.to_csv(fname)
    print(fname)
    return finaldataf


#parcorr = dataframeextract('Partial correlation')
#graph_covar = dataframeextract('GraphLassoCV covariance')
#graph_precis = dataframeextract('GraphLassoCV precision')
#covar = dataframeextract('Covariance')
corr = dataframeextract('Correlation')
print('Debug')
