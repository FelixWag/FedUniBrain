########################################################################################
# Code taken from our previous paper: https://github.com/WenTXuL/MultiUnet
########################################################################################
import preprocess
import os

if __name__ == "__main__":
    ########################################################################################
    #'data_path' should point to the folder with the file that contains mulit_class labels 
    #'save_path' should point to the folder that saves the merged label
    ########################################################################################
   
    data_path = "/datapath/BRATS_label"
    save_path = "/savepath/BRATS_label_merged"

    # labels
	# "0": "background", 
	# "1": "edema",
	# "2": "non-enhancing tumor",
	# "3": "enhancing tumour"
    
    load_name_list=os.listdir(data_path)
    load_name_list=sorted(load_name_list)

    for load_name in load_name_list:        
        print (load_name)                
        save_name = load_name.split('.')[0]+ "_merged.nii.gz"
        nifti_array,affine= preprocess.load_nifti(data_path=data_path, file_name=load_name)   
        nifti_array[nifti_array==1] = 1
        nifti_array[nifti_array==2] = 1
        nifti_array[nifti_array==3] = 1
        preprocess.save_arr(nifti_array,save_path,save_name, affine)
    
    