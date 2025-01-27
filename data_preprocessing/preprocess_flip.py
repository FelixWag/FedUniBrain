########################################################################################
# Code taken from our previous paper: https://github.com/WenTXuL/MultiUnet
########################################################################################

import nibabel as nib
import os
import numpy as np
import numpy.ma as ma
from glob import glob
import preprocess

if __name__ == "__main__":
    
    ##################################################################################################################
    # To use this code for preprocessing, you need a file structure:  
    #         /Databasesname/Samplename/Images_or_Labels/file
    # Databasesname: the databases folder
    # Samplename: each MRI data sample (all the modalities of that sample are included in this folder )
    # Images_or_Labels: two folders under the sample folder: Images , Labels 
    # file: The Images folder includes files with different modalities, showing names such as modalities.nii.gz (e.g. FLAIR.nii.gz). You can also put the brain mask in this file if you have as (MASK.nii.gz) For the Labels folder: it only contains one file (label.nii.gz)
    # For saving the preprocessed data, you need a file structure as 
    # /Databasesname/Images_or_Labels
    # Then, you need to change the setting in the code
    # 1. change the Data_set_path to the location of your database
    # 2. change the modality name 
    # 3. change the save path to the location that you want for the preprocessed data
    # The intensity normalization methods we used need brain masks. You can use the brain masks you get from skull strip tools or the database. If you can't get you can use the get brain mask methods 
    # This code is similar to preprocess.py but it also flip the images
    ##################################################################################################################

    data_input_path="/path/databasename" 
    data_save_path = "/path/databasename" 
    modality_names = ["FLAIR", "T1", "GADO", "T2", "DP"]   #example on MSSEG
    subsets = os.listdir(data_input_path)
    subsets.sort()
    #Assume each sample is stored in a specific folder 
    for subset in subsets:
        if not os.path.isdir(os.path.join(data_input_path, subset)):          
            continue         
        subset_path = os.path.join(data_input_path, subset)
        img_path=os.path.join(subset_path, "Images")
        seg_path=os.path.join(subset_path, "Labels")
        #Assume each modality_file is named as modality.nii.gz              
        print(subset)        
        modalities_array = []
        affine_array = []       
        normed_array = [] 
        for modality_name in modality_names:

            modality_file = os.path.join(img_path, modality_name+".nii.gz")
            modality_nifti= nib.load(modality_file)            
            modality_arr = preprocess.nifti_to_array(modality_nifti)
            img_affine=modality_nifti.affine
            #!!!here, you can use the mask file from the databases or the skull strip tool. Assume it is under the Images folder and called MASK.nii.gz
            if os.path.isfile(os.path.join(img_path,"MASK.nii.gz")) :
                modality_mask = nib.load(os.path.join(img_path,"MASK.nii.gz")) 
                modality_mask = preprocess.nifti_to_array(modality_mask)
                modality_mask = modality_mask > 0.5
            #!!!If you don't have the mask file, the following code will generate one
            else:
                modality_mask = modality_arr != 0  #need to change if the background intensity is nort zero
                modality_mask = modality_mask.astype(bool)

            modality_normed = preprocess.normalise(modality_arr, modality_mask)
            normed_array.append(modality_normed)

        concat_modalities = np.stack(normed_array, axis = -1)       
        seg_file = nib.load(os.path.join(seg_path,"label.nii.gz"))
        seg_arr = preprocess.nifti_to_array(seg_file)
        seg_affine=seg_file.affine
        save_name = subset + ".nii.gz"
        img_save_path = os.path.join(data_save_path,"Images" )
        img_save_path = os.path.join(img_save_path, save_name)
        label_save_path =os.path.join(data_save_path, "Labels")
        label_save_path =os.path.join(label_save_path, save_name)

        #Flip  You may need this to change the orientation of the brain
        concat_modalities=np.flip(concat_modalities, (0, 1))
        seg_arr=np.flip(seg_arr,(0,1))
        rot=np.identity(4)
        rot[0,0]=-1
        rot[1,1]=-1
        img_affine=np.dot(img_affine,rot)
        seg_affine=np.dot(seg_affine,rot)          
        img_file = nib.Nifti1Image(concat_modalities, affine=img_affine)
        seg_file_from_arr = nib.Nifti1Image(seg_arr, affine=seg_affine) 
        nib.save(img_file,img_save_path)
        nib.save(seg_file_from_arr,label_save_path)

        