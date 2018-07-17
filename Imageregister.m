dirpar = '/projects/perinatal/peridata/EPRIME/';
listtempsph = dir(dirpar);
dirout = '/home/sbh17/Documents/KURFEprime/';

for i = 1:length(listtempsph)
    if (length(listtempsph(i).name) == 6) && ~strcmpi(listtempsph(i).name,'README')
        cmd = ['fsl5.0-flirt -in ' dirpar listtempsph(i).name '/003_volumetric_data/' listtempsph(i).name '_brain.nii.gz -ref ' dirpar listtempsph(i).name '/003_volumetric_data/' listtempsph(i).name '_brain.nii.gz -out /home/sbh17/Documents/KURFEprime2/' listtempsph(i).name '_brain.nii.gz -applyisoxfm 2'];
        cmd2 = ['fsl5.0-flirt -in ' dirpar listtempsph(i).name '/003_volumetric_data/segmentations/' listtempsph(i).name '_all_labels.nii.gz -ref /home/sbh17/Documents/KURFEprime2/' listtempsph(i).name '_brain.nii.gz -out /home/sbh17/Documents/KURFEprime2/' listtempsph(i).name '_all_labels.nii.gz'];
        cmd3 = ['fsl5.0-flirt -in ' dirpar listtempsph(i).name '/002_functional_data/' listtempsph(i).name '_rs_fMRI.ica/mean_func.nii.gz -ref /home/sbh17/Documents/KURFEprime2/' listtempsph(i).name '_brain.nii.gz -out /home/sbh17/Documents/KURFEprime2/' listtempsph(i).name '_EPI_T2.nii.gz -omat /home/sbh17/Documents/KURFEprime2/' listtempsph(i).name '_EPI_T2.mat -dof 6'];
        cmd4 = ['fsl5.0-flirt -ref /home/sbh17/Documents/KURFEprime2/' listtempsph(i).name '_brain.nii.gz -in ' dirpar listtempsph(i).name '/002_functional_data/' listtempsph(i).name '_clean_rs_fMRI.nii.gz -applyxfm -init /home/sbh17/Documents/KURFEprime2/' listtempsph(i).name '_EPI_T2.mat  -out  /home/sbh17/Documents/KURFEprime2/' listtempsph(i).name '_clean_rest_to_T2.nii.gz -dof 6'];
        a = system(cmd);
        b = system(cmd2);
        c = system(cmd3);
        d = system(cmd4);
        if (a ~= 0) || (b~= 0) || (c ~= 0) || (d ~= 0)
            eval(['delete /home/sbh17/Documents/KURFEprime2/' listtempsph(i).name '_all_labels.nii.gz'])
            eval(['delete /home/sbh17/Documents/KURFEprime2/' listtempsph(i).name '_brain.nii.gz'])
        end
    end
    
end

