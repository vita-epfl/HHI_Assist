import os
import glob 
import numpy as np
from torch.utils.data import Dataset
from utils.misc import get_info
from utils.data_customing import read_data
import pandas as pd
import random
import torch

class HHI(Dataset):
    def __init__(self, split, **kwargs):
        """
        :param path_to_data
        :param input_nbr_frames
        :param output_nbr_frames
        :param task_chosen
        :param type of data : train 0 validation 1 test 2
        """
        data_dir = kwargs['data_dir']
        input_n = kwargs['input_n']
        output_n = kwargs['output_n']
        sample_rate = kwargs['sample_rate']
        ov_factor = kwargs['ov']
        h = kwargs['h']
        self.path_to_data = os.path.join(data_dir)
        self.in_n = input_n
        self.out_n = output_n 
        self.p3d = {}
        self.data_idx = []
        seq_len = self.in_n + self.out_n 

        data_dir = "."
        # Key for indexing data elements as loaded

        subjects = [["AA-RM","AB-JB","BC-CC","BC-HKG","BC-JH","SL-JH","BC-LPR","ED-GG","ED-MT","JB-BC","RM-ED","SR-LPR","SL-LPR","PH-BC", "ED-EC"],["SC-PH"],["GR-JW","OP-MF","KT-MA"]]

         
        print(subjects)
        if split == 0: # training
            subs = subjects[0]
        elif split == 1: # validating
            subs = subjects[1]
        elif split == 2: # testing
            subs = subjects[2]

        key = 0
        for sub_folder in os.listdir(data_dir):
            if sub_folder in subs:
                print("subfolder", sub_folder)
                data_path = os.path.join(data_dir, sub_folder)
                files = [f for f in os.listdir(data_path) if f.endswith(".csv")]
                files.sort()

                for i in range(0, len(files), 2):
                    csv_path = os.path.join(data_dir+"/"+sub_folder+"/", files[i])
                    csv_path2 = os.path.join(data_dir+"/"+sub_folder+"/", files[i+1])
                    # breakpoint()

                    p1 = get_info(csv_path, sub_folder)
                    p2 = get_info(csv_path2, sub_folder)

                    # ORDER IS CARE RECEIVER FIRST, CARE GIVER SECOND

                    # Read CSV File
                    if p1 == "CR" and p2 == "CG":
                        df = pd.read_csv(csv_path)
                        df2 = pd.read_csv(csv_path2)
                    elif p1 == "CG" and p2 == "CR":
                        df = pd.read_csv(csv_path2)
                        df2 = pd.read_csv(csv_path)

                    print(f"Processing both files: {csv_path} and {csv_path2}")
                    try:
                        df, df2 = read_data(df, df2, sample_rate, h)
                    except:
                        breakpoint()


                    nr_samples = (int(df.shape[0] / 24) - 2) * ov_factor + 1
                    print("nr_samples: ", nr_samples)

                    # FINAL ORDER TURNS OUT TO BE CARE GIVER THEN CARE RECEIVER 
                    df2 = df2.to_numpy()
                    w1, w2 = df2.shape 
                    df2 = df2.reshape(w1, int(w2/3), 3)
                    df = df.to_numpy().reshape(w1, int(w2/3), 3)
                    both = np.concatenate((df2, df), 2).reshape(w1, 2*w2)

                    if (nr_samples >= 1):
                        npped = both
                        augment1 = [key] * nr_samples
                        augment2 = [int(self.in_n/ov_factor) * i for i in range(nr_samples)]
                        print("Shape of npped")
                        print(npped.shape)

                        # npped shape is (nr of frame, nr of joints * 3)
                        self.p3d[key] = npped

                        self.data_idx.extend(zip(augment1, augment2))
                        key += nr_samples

        print("length is ",np.shape(self.data_idx)[0])
    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)
        
        pose = self.p3d[key][fs]

        mask = np.zeros((pose.shape[0], pose.shape[1]))
        mask[0:self.in_n, :] = 1

        mask[self.in_n:self.in_n + self.out_n, :] = 0

        data = {
            "pose": pose,
            "mask": mask,
            "timepoints": np.arange(self.in_n + self.out_n)
        }

        return data