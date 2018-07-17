# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 13:35:00 2018

@author: surya
"""

import os
import nibabel as nib
import numpy as np
import pandas as pd



#Uncomment on the work PC:
#'''
Directory1 = "/projects/perinatal/peridata/EPRIME/"
Directory2 = "/home/sbh17/Documents/KURFEprime2"
#'''

#Uncomment on local PC:
'''
Directory1 = "C:\\Users\\surya\\OneDrive - King's College London\\KURF\\Data2"
Directory2 = "C:\\Users\\surya\\OneDrive - King's College London\\KURF\\Data2"
#'''



def labelextractor(sub_id):
    excl_list=[]
    count = 0
    for i in range (0,len(sub_id.iloc[:])):
        label_file = 'EP' + str(np.int(sub_id.iloc[i])) + '_all_labels.nii.gz'
        Directorylabel = Directory1 + 'EP' + str(np.int(sub_id.iloc[i])) + '/003_volumetric_data/segmentations/'
        labelpath = os.path.join(Directorylabel,label_file)
        try:
            datalabel = np.unique(nib.load(labelpath).get_fdata())
        except FileNotFoundError:
            message = 'Data for ' + 'EP' + str(np.int(sub_id.iloc[i])) + ' missing.'
            excl_list.append('EP' + str(np.int(sub_id.iloc[i])))
            print(message)
        else: 
            if len(datalabel)==88:
                out = datalabel
                count = count+1
            else:
                message2 = 'Data for ' + 'EP' + str(np.int(sub_id.iloc[i])) + ' not used.'
                print(message2)
                excl_list.append('EP' + str(np.int(sub_id.iloc[i])))
    return out,excl_list,count


def meantimes(data_arr,datalabel,labels):
    
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
                out[i,np.int(j)-1]=np.mean(time_arr[np.where(datalabel[:,:,:,0]==j)])
            except:
                out[np.int(j)-1]=np.mean(time_arr[np.where(datalabel[:,:,:,0]==j)])
    return out


        

def dataextractor(subject,labels):
    Directoryfile = Directory1 + subject + '/003_volumetric_data/'
    Directorylabel = Directory1 + subject + '/003_volumetric_data/segmentations/'
    volume_file = subject + '_T2.nii.gz'
    filepath = os.path.join(Directoryfile,volume_file)
    label_file = subject + '_all_labels.nii.gz'            
    labelpath = os.path.join(Directorylabel,label_file)
    try:
        dataarr = nib.load(filepath).get_fdata()
        datalabel = nib.load(labelpath).get_fdata()
    except FileNotFoundError:
        message = 'Data for ' + subject + ' missing.'
        print(message)
        return
    else:
        time_avg = meantimes(dataarr,datalabel,labels)
        return time_avg.flatten()
        


def dataframeextract(typedat):
    file_sub_csv = os.path.join(Directory2,'eprime_INFO.csv')
    subs = pd.read_csv(file_sub_csv, sep = ',')
    sub_id = subs.iloc[:,0]
    sub_id = sub_id.dropna(axis=0)
    labels,excl_list,count = labelextractor(sub_id)
    ind1 = len(labels)-1
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
            corr_cov_mat = dataextractor(subject,labels)
            feat_vect = corr_cov_mat
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





data = dataframeextract('Volumes_means')
