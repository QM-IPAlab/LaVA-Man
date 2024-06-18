import math
import sys
import torch
from cliport.models.core import attention
import util.misc as misc
from mae.engine_pretrain_ours import generate_token
import h5py
import os
import numpy as np
from tqdm import tqdm
TESTSET_IDX = [0,716,1216,1716,2205,2694,2928,3205,3305,3535,3757,3924,4091,5015,5933,6570,7207,7920,8633]
import time

def save_relevance_maps(model, 
                        data_loader_train, 
                        data_loader_vis, 
                        device, 
                        args, 
                        text_processor):
    
    def run(data_loader, name):

        model.train(True)
        relevance_data = []
        input_ids_data = []
        attention_mask_data = []
        num = 0
        
        for batch in tqdm(data_loader):
            img1, img2, lang, pick, place = batch
            img1 = img1.to(device, non_blocking=True).half()
            img2 = img2.to(device, non_blocking=True).half()
            pick = pick.to(device, non_blocking=True).half()
            place = place.to(device, non_blocking=True).half()
            # put the tokenizer here to avoid deadlock caused by the fork of the tokenizer
           
            processed_lang = generate_token(text_processor, lang, device)       
            with torch.cuda.amp.autocast():
                relevance_map = model.show_relevance_map(img1, processed_lang)
            # relevance map : b, 1, h, w
            
            relevance_data.append(relevance_map.cpu().numpy())
            input_ids_data.append(processed_lang['input_ids'].cpu().numpy())
            attention_mask_data.append(processed_lang['attention_mask'].cpu().numpy())
            num += 1
            
            torch.cuda.empty_cache()

        relevance_data = np.concatenate(relevance_data, axis=0)
        input_ids_data = np.concatenate(input_ids_data, axis=0)
        attention_mask_data = attention_mask_data = np.concatenate(attention_mask_data, axis=0)

        f = h5py.File(os.path.join('/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/data_hdf5',
                                    f'{name}'), 'a')

        existing_length = len(f['image_s1']) if 'image_s1' in f else 0
        
        if existing_length == len(relevance_data) and existing_length == len(input_ids_data):
            n6 = append_or_create_dataset(f, 'relevance_map', data=relevance_data)
            n7 = append_or_create_dataset(f, 'input_ids', data=input_ids_data)
            n8 = append_or_create_dataset(f, 'attention_mask', data=attention_mask_data)
            f.close()
            print("Relevance maps saved successfully")
            print("Length of the dataset: ", n6)
            print("Length of the dataset: ", n7)
            print("Length of the dataset: ", n8)
        else:
            f.close()
        raise ValueError("Length of data does not match existing length in the dataset")

    run(data_loader_train, 'exist_dataset_no_aug_all.hdf5')
    run(data_loader_vis, 'exist_dataset_no_aug_all_test.hdf5')


def append_or_create_dataset(f, name, data, dtype=None):

    if name in f:
        # If dataset already exists, append to it
        dset = f[name]
        dset.resize(dset.shape[0] + len(data), axis=0)
        dset[-len(data):] = data
        n = len(dset)

    else:

        if dtype is None:
            maxshape = (None,) + data[0].shape
            chunks = (1,) + data[0].shape
            f.create_dataset(name, data=data, maxshape=maxshape,
                                chunks=chunks)
        else:
            maxshape = (None,)
            chunks = (1,)  # For variable-length data
            f.create_dataset(name, data=data, maxshape=maxshape,
                                chunks=chunks, dtype=dtype)

        n = len(data)

    return n