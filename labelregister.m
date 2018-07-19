dirpar = '/projects/perinatal/peridata/EPRIME/';
listtempsph = dir(dirpar);
dirout = '/home/sbh17/Documents/KURFEprime/';

for i = 1:length(listtempsph)
    if (length(listtempsph(i).name) == 6) && ~strcmpi(listtempsph(i).name,'README')
        if exist(['/home/sbh17/Documents/KURFEprime2/' listtempsph(i).name '_all_labels.nii.gz'], 'file')==2
            cmd = ['fsl5.0-flirt -in ' dirpar listtempsph(i).name '/003_volumetric_data/segmentations/' listtempsph(i).name '_all_labels.nii.gz -ref ' dirpar listtempsph(i).name '/003_volumetric_data/' listtempsph(i).name '_brain.nii.gz -out /home/sbh17/Documents/KURFEprime2/' listtempsph(i).name '_all_labels.nii.gz -applyisoxfm 2 -interp nearestneighbour'];
            system(cmd)
        end
    end
end