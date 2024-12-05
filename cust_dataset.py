import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
import torch
import glob
from torchvision import transforms
import random
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.discriminant_analysis import StandardScaler



class CustomDataset(Dataset):
    
    def __init__(self, img_path='/data/neuromark2/Data/ABCD/Data_BIDS_5/Raw_Data/',
                 label_file='CBCL_filtered_subjects_updated.xlsx', transform=None,
                 target_transform=None, train=True, valid=False, random_state=42):
        path = '/data/users3/rmandepudi1/final/CBCL'
        print("Initializing CustomDataset 1")
        self.img_path = img_path
        self.dirs = os.listdir(img_path)
        mask_path = '/home/users/rmandepudi1/MaskResult0.2.nii'
        # mask_cfp = '/data/users3/rmandepudi1/final/masks/CFP_FMask.nii'
        self.mask = nib.load(mask_path).get_fdata()
       

        row_values = 11734
        self.vars = pd.read_excel(os.path.join(path, label_file), index_col='src_subject_id',
                          usecols=['src_subject_id', 'cbcl_scr_syn_attention_r'], nrows=row_values)

        self.num_selected_rows = len(self.vars)
        # print("self.vars", self.vars)
        print("No of rows Selected" , self.num_selected_rows)
        self.vars.columns = ['cbcl_scr_syn_attention_r']
        num_nans_original = self.vars['cbcl_scr_syn_attention_r'].isna().sum()

      
        # self.vars['new_score'] = self.vars['tfmri_nb_all_beh_ctotal_mrt']
        data_to_scale = np.array(self.vars['cbcl_scr_syn_attention_r']).reshape(-1, 1)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_to_scale)
        self.vars['new_score'] = scaled_data.ravel()
        num_nans = self.vars['new_score'].isna().sum()

        print("Loaded labels...") 
        # sss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        # print("sss done")
        # self.train_idx, self.test_idx = next(sss.split(np.zeros_like(self.vars),
        #                                            self.vars.new_score.values))
        self.train_idx = list(range(int(0.8 * row_values)))
        self.test_idx = list(range(int(0.8 * row_values), row_values))
        # if train or valid:
        #     self.vars = self.vars.iloc[train_idx]
        # else:
        #     test_vars = self.vars.iloc[self.test_idx]
        
        self.vars = self.vars.sort_index()    
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.missing_dirs = []
        # self.check_for_missing_data()
        # self.print_missing_dirs()
        

        print("CustomDataset initialized.")

    
    def print_new_score_range(self):
        """Function to print the range of new_score"""
        min_score = self.vars['new_score'].min()
        max_score = self.vars['new_score'].max()
        print(f"Range of 'new_score' - Min: {min_score}, Max: {max_score}")

        
    def __len__(self):
        return len(self.vars)

    # def __getitem__(self, idx):
    #     subject_dir = os.path.join(self.img_path, self.vars.index[idx])
    #     # Check if the subject directory exists
    #     if not os.path.exists(subject_dir):
    #         raise FileNotFoundError(f"Subject directory {subject_dir} not found.")
        
    #     # Specify the path to the target file
    #     target_file_path = os.path.join(subject_dir, 'Baseline', 'dti', 'dti_FA', 'tbdti32ch_FA.nii.gz')
        
    #     # Check if the target file exists
    #     if not os.path.exists(target_file_path):
    #         raise FileNotFoundError(f"Target file {target_file_path} not found.")
        
    #     # Load the data
    #     img = nib.load(target_file_path).get_fdata()

 
    def __getitem__(self, idx):
        subject_dir = os.path.join(self.img_path, self.vars.index[idx])
        # Check if the subject directory exists
        if not os.path.exists(subject_dir):
            self.missing_dirs.append(subject_dir)
            return None 
            # raise FileNotFoundError(f"Subject directory {subject_dir} not found.")

        # Define the Baseline directory
        baseline_dir = os.path.join(subject_dir, 'Baseline')

        # Pattern match for the anat_2... or anat_NORM_2... directories
        patterns = ['anat_2*', 'anat_NORM_2*']
        target_file_path = None

        for pattern in patterns:
            # Construct the search path for directories matching the pattern
            search_path = os.path.join(baseline_dir, pattern)

            # Use glob to find all matching directories
            matching_dirs = glob.glob(search_path)

            # Check each matching directory for the target file
            for dir_path in matching_dirs:
                potential_file_path = os.path.join(dir_path, 'smwc1pT1.nii')
                if os.path.exists(potential_file_path):
                    target_file_path = potential_file_path
                    break  # Stop if the file is found

            if target_file_path:
                break  # Stop if the file is found in any of the pattern matched directories

        # Check if the target file was found
        if not target_file_path:
            print('Miss')
            self.missing_dirs.append(f"Target file missing for {subject_dir}")

            return None 
            # raise FileNotFoundError("Target file smwc1pT1.nii not found in any expected directory.")

        # Load the data
        img = nib.load(target_file_path).get_fdata()
        label = self.vars.iloc[idx]

        # Preprocess image data
        img = torch.tensor(img, dtype=torch.float)
        # img = (img - img.mean())/img.std()   # Normalizing the image
        img = (img - img.mean())
        img = img * self.mask  # Assuming self.mask is defined and correctly shaped
    

        if torch.sum(torch.isnan(img)) > 0:
            print(f'Custom dataset, {idx}')
            exit(-1)

        # Apply transformations
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img,(label['new_score'])
    
    def check_for_missing_data(self):
        """
        Checks for missing subject directories and prints them.
        """
        print("Checking for missing subject directories...")
        for i in range(len(self)):
            self.__getitem__(i)

        self.print_missing_dirs()

    def print_missing_dirs(self):
        """
        Prints missing directories collected during data loading.
        """
        if self.missing_dirs:
            print("The following subject directories were not found:")
            for dir in self.missing_dirs:
                print(dir)
