
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
        try:
            data=data_arr[:,:,:,i]
        except:
            data = data_arr
        for j in labels[1:]:
            time_arr = data
            try:
                out[i,np.int(j)-1]=np.mean(time_arr[np.where(datalabel==j)])
            except:
                out[i,np.int(j)-1]=np.mean(time_arr[np.where(datalabel[:,:,:,0]==j)])
    return out

def extractupper(arr):
    out = np.array([])
    for i in range(0, arr.shape[0]):
        out = np.concatenate([out, arr[i, i+1:]])
    return out
        

def dataextractor(subject,labels):
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
        fname = '/home/sbh17/Documents/KURFEprime2/' + subject + '.txt'
        np.savetxt(fname, time_avg, delimiter=',')
        return time_avg
        

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

file_sub_csv = os.path.join(Directory2,'eprime_INFO.csv')
subs = pd.read_csv(file_sub_csv, sep = ',')
sub_id = subs.iloc[:,0]
sub_id = sub_id.dropna(axis=0)
labels,excl_list,count = labelextractor(sub_id)

for i in range (0,len(sub_id.iloc[:])):
    #obtain file and pathname:
    subject = 'EP' + str(np.int(sub_id.iloc[i]))
    if subject not in excl_list:
        dataextractor(subject,labels)
        message = subject + ' complete.'
        print(message)
    else:
        continue

