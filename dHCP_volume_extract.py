# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 16:00:36 2018

@author: surya
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 15:48:18 2018

@author: surya
"""


import os
import nibabel as nib
import numpy as np
import pandas as pd
#import matlab.engine
import matplotlib.pyplot as plt
from scipy import stats, linalg
from sklearn.covariance import GraphLassoCV
from itertools import product
from functools import reduce
#import nilearn
import glob



#Uncomment on the work PC:
#'''
Directory1 = "/projects/perinatal/peridata/dHCP/"
Directory2 = "/home/sbh17/"
#'''

#Uncomment on local PC:
'''
Directory1 = "C:\\Users\\surya\\OneDrive - King's College London\\KURF\\Data2"
Directory2 = "C:\\Users\\surya\\OneDrive - King's College London\\KURF\\Data2"
#'''

def labelextractor():
    excl_list=[]
    count = 0
    label_names = glob.glob(os.path.join("/projects/perinatal/peridata/dHCP/labels",'*_drawem_all_labels.nii.gz'))
    for i in range (0,len(label_names)):
        labelpath = label_names[i]
        try:
            datalabel = np.unique(nib.load(labelpath).get_fdata())
        except FileNotFoundError:
            message = 'Data for ' + labelpath[labelpath.find('CC0'):labelpath.find('_drawem')] + ' missing.'
            excl_list.append(labelpath[labelpath.find('CC0'):labelpath.find('_drawem')])
            print(message)
        else: 
            if len(datalabel)==88:
                out = datalabel
                count = count+1
            else:
                message2 = 'Data for ' + labelpath[labelpath.find('CC0'):labelpath.find('_drawem')] + ' not used.'
                print(message2)
                excl_list.append(labelpath[labelpath.find('CC0'):labelpath.find('_drawem')])
        print('Labels for ' + labelpath[labelpath.find('CC0'):labelpath.find('_drawem')] + ' investigated.')
    return out,excl_list,count


def meantimes(data_arr,datalabel,labels,dimensions):
    try:
        t = data_arr.shape[3]
    except:
        t = 1 
    out = np.zeros([t,len(labels)-1])
    for i in range(0,t):
        try:
            data=data_arr[:,:,:,i]
        except:
            data = data_arr
        for j in labels[1:]:
            time_arr = data
            try:
                out[i,np.int(j)-1]=len(time_arr[np.where(datalabel==j)])*reduce(lambda x, y: x*y,
                    dimensions)
            except:
                out[i,np.int(j)-1]=len(time_arr[np.where(datalabel[:,:,:,0]==j)])*reduce(lambda x, y: x*y,
                    dimensions)           
    return out

def dataextractor(subject,session,labels):
    T2w_nii = subject + '_' + session + '_T2w_restore.nii.gz'
    filepath = os.path.join(Directory1,'T2',T2w_nii)
    label_file = subject + '_' + session + '_drawem_all_labels.nii.gz'            
    labelpath = os.path.join(Directory1,'labels',label_file)
    try:
        dataarr = nib.load(filepath).get_fdata()
        datalabel = nib.load(labelpath).get_fdata()
    except FileNotFoundError:
        message = 'Data for ' + subject + ' missing.'
        print(message)
        return
    else:
        T2_dimensions = np.divide(nib.load(filepath).header.get_zooms(),100)
        volumes = meantimes(dataarr,datalabel,labels,T2_dimensions)
        return volumes.flatten()


def dataframeextract(typedat):
    file_sub_csv = os.path.join(Directory2,'dHCP_info.csv')
    subs = pd.read_csv(file_sub_csv, sep = ',')
    sub_id = subs['Subject']
    sub_id = sub_id.dropna(axis=0)
    ses_no = subs['Session']
    ses_no = ses_no.dropna(axis=0)
    labels,excl_list,count = labelextractor()
    ind1 = len(labels)-1
    feat_set = []
    sub_id_final = []
    ses_no_final = []
    ind = 0
    gender = []
    gest_age = []
    scan_age = []
    print(typedat)
    for i in range (0,len(sub_id.iloc[:])):
        #obtain file and pathname:
        subject = 'sub-' + sub_id[i]
        session = 'ses-' + str(ses_no[i])
        if subject not in excl_list:
            volumes = dataextractor(subject,session,labels)
            try:
                len(volumes)                
            except:
                print('Data missing for ' + subject + ' in session ' + session + '.')
                continue
            else:
                feat_vect = volumes
                feat_set.append(feat_vect)
                sub_id_final.append(subject)
                ses_no_final.append(ses_no[i])
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
                            'Session' : ses_no_final,
                            'Gestation age' : gest_age,
                            'Age at scan': scan_age})
    labelsdata = np.arange(0,ind1)
    labelsdata = list(str(i) for i in labelsdata)
    datafromsubs = pd.DataFrame(data=np.array(feat_set),columns=labelsdata)
    finaldataf = pd.concat([Attributes, datafromsubs], axis=1)
    index = list(str(i) for i in range(0, len(sub_id_final)))
    finaldataf = finaldataf.set_index([index, 'Subject ID'])
    fname = typedat + '.csv'
    finaldataf.to_csv(fname)
    print(fname)
    return finaldataf


data = dataframeextract('dHCP_Volumes')

