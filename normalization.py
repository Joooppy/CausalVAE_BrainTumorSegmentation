"""
Adapted from:
@author: Chenggang
@github: https://github.com/MissShihongHowRU
@time: 2020-09-09 20:36
"""
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Pool
import os
import time
import pandas as pd
import argparse
from config import config

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--data_path', type=str, help='The data path')
    parser.add_argument('-y', '--year', type=int, help='s', default=2020)
    return parser.parse_args()

def get_list_of_files(base_dir):
    list_of_lists = []
    for glioma_type in ['HGG', 'LGG']:
        current_directory = join(base_dir, glioma_type)
        patients = subfolders(current_directory, join=False)
        for p in patients:
            patient_directory = join(current_directory, p)
            t1_file = join(patient_directory, p + "_t1.nii")
            t1c_file = join(patient_directory, p + "_t1ce.nii")
            t2_file = join(patient_directory, p + "_t2.nii")
            flair_file = join(patient_directory, p + "_flair.nii")
            seg_file = join(patient_directory, p + "_seg.nii")
            this_case = [t1_file, t1c_file, t2_file, flair_file, seg_file]
            assert all((isfile(i) for i in this_case)), f"some file is missing for patient {p}; make sure the following files are there: {this_case}"
            list_of_lists.append(this_case)
    print(f"Found {len(list_of_lists)} patients")
    return list_of_lists

def get_list_of_files_2020(current_directory, patients, mode="training"):
    list_of_lists = []
    for p in patients:
        patient_directory = join(current_directory, p)
        t1_file = join(patient_directory, p + "_t1.nii")
        t1c_file = join(patient_directory, p + "_t1ce.nii")
        t2_file = join(patient_directory, p + "_t2.nii")
        flair_file = join(patient_directory, p + "_flair.nii")
        if mode == "training":
            seg_file = join(patient_directory, p + "_seg.nii")
            this_case = [t1_file, t1c_file, t2_file, flair_file, seg_file]
        else:
            this_case = [t1_file, t1c_file, t2_file, flair_file]
        assert all((isfile(i) for i in this_case)), f"some file is missing for patient {p}; make sure the following files are there: {this_case}"
        list_of_lists.append(this_case)
    print(f"Found {len(list_of_lists)} patients")
    return list_of_lists

def load_and_preprocess(case, patient_name, output_folder):
    imgs_sitk = [sitk.ReadImage(i) for i in case]
    imgs_npy = [sitk.GetArrayFromImage(i) for i in imgs_sitk]
    imgs_npy = np.concatenate([i[None] for i in imgs_npy]).astype(np.float32)
    nonzero_masks = [i != 0 for i in imgs_npy[:-1]]
    brain_mask = np.zeros(imgs_npy.shape[1:], dtype=bool)
    for i in range(len(nonzero_masks)):
        brain_mask = brain_mask | nonzero_masks[i]
    for i in range(len(imgs_npy) - 1):
        mean = imgs_npy[i][brain_mask].mean()
        std = imgs_npy[i][brain_mask].std()
        imgs_npy[i] = (imgs_npy[i] - mean) / (std + 1e-8)
        imgs_npy[i][brain_mask == 0] = 0
    os.makedirs(output_folder, exist_ok=True)
    np.save(join(output_folder, patient_name + ".npy"), imgs_npy)
    print(f"Processed and saved: {patient_name}")

def load_and_preprocess_val(case, patient_name, output_folder):
    imgs_nib = [nib.load(i) for i in case]
    imgs_sitk = [sitk.ReadImage(i) for i in case]
    imgs_npy = [sitk.GetArrayFromImage(i) for i in imgs_sitk]
    affines = [i.affine for i in imgs_nib]
    imgs_npy = np.concatenate([i[None] for i in imgs_npy]).astype(np.float32)
    affines = np.concatenate([i[None] for i in affines]).astype(np.float32)
    nonzero_masks = [i != 0 for i in imgs_npy]
    brain_mask = np.zeros(imgs_npy.shape[1:], dtype=bool)
    for i in range(len(nonzero_masks)):
        brain_mask = brain_mask | nonzero_masks[i]
    for i in range(len(imgs_npy)):
        mean = imgs_npy[i][brain_mask].mean()
        std = imgs_npy[i][brain_mask].std()
        imgs_npy[i] = (imgs_npy[i] - mean) / (std + 1e-8)
        imgs_npy[i][brain_mask == 0] = 0
    affine_output_folder = output_folder[:-4] + '/affine'
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(affine_output_folder, exist_ok=True)
    np.save(join(output_folder, patient_name + ".npy"), imgs_npy)
    # np.save(join(affine_output_folder, patient_name + ".npy"), affines)
    print(f"Processed and saved: {patient_name}")

if __name__ == "__main__":
    args = init_args()
    if args.year == 2018:
        data_file_path = "data/MICCAI_BraTS_2018_Data_Training"
        npy_normalized_folder = join(data_file_path, "npy")
        list_of_lists = get_list_of_files(data_file_path)
        patient_names = [i[0].split(os.sep)[-2] for i in list_of_lists]
        p = Pool(processes=8)
        t0 = time.time()
        print("Job starts")
        p.starmap(load_and_preprocess, zip(list_of_lists, patient_names, [npy_normalized_folder] * len(list_of_lists)))
        print(f"Finished; costs {time.time() - t0}s")
        p.close()
        p.join()
    if args.year == 2020:
        data_file_path = "data/MICCAI_BraTS2020_TrainingData"
        npy_normalized_folder = join(data_file_path, "npy")
        mapping_file_path = join(data_file_path, "name_mapping.csv")
        name_mapping = pd.read_csv(mapping_file_path)
        HGG = name_mapping.loc[name_mapping.Grade == "HGG", "BraTS_2020_subject_ID"].tolist()
        LGG = name_mapping.loc[name_mapping.Grade == "LGG", "BraTS_2020_subject_ID"].tolist()
        patients = HGG + LGG
        list_of_lists = get_list_of_files_2020(data_file_path, patients)
        p = Pool(processes=8)
        t0 = time.time()
        print("Job starts")
        p.starmap(load_and_preprocess, zip(list_of_lists, patients, [npy_normalized_folder] * len(list_of_lists)))
        print(f"Finished; costs {time.time() - t0}s")
        p.close()
        p.join()
    if args.year in [202001, 202002]:
        if args.year == 202001:
            data_file_path = "data/MICCAI_BraTS2020_ValidationData"
            mapping_file_path = join(data_file_path, "name_mapping_validation_data.csv")
        if args.year == 202002:
            data_file_path = join(config["base_path"], "MICCAI_BraTS2020_TestingData")
            mapping_file_path = join(data_file_path, "survival_evaluation.csv")
        npy_normalized_folder = join(data_file_path, "npy")
        name_mapping = pd.read_csv(mapping_file_path)
        patients = name_mapping["BraTS20ID"].tolist()
        list_of_lists = get_list_of_files_2020(data_file_path, patients, mode="validation")
        p = Pool(processes=8)
        t0 = time.time()
        print("Job starts")
        p.starmap(load_and_preprocess_val, zip(list_of_lists, patients, [npy_normalized_folder] * len(list_of_lists)))
        print(f"Finished; costs {time.time() - t0}s")
        p.close()
        p.join()