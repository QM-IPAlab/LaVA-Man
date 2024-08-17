"""
Save existing dataset to hdf5 format
"""

import h5py
import numpy as np
# file = h5py.File(os.path.join('/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/data_hdf5',
#                               'exist_dataset_no_aug.hdf5'), 'r+')

# img1 = file['image_s1'][:13387]
# img2 = file['image_s2'][:13387]
# lang = file['language'][:13387]
# pick = file['gt_pick'][:13387]
# place = file['gt_place'][:13387]


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


def overwrite_dataset(f, name, data, dtype=None):
    if name in f:
        del f[name]
    if dtype is not None:
        f.create_dataset(name, data=data, dtype=dtype)
    else:
        f.create_dataset(name, data=data)

def merge_datasets(file1, file2, output_file):
    """
    To merget two hdf5 files

    Args:
    file1: str, path to the first hdf5 file
    file2: str, path to the second hdf5 file
    output_file: str, path to the output hdf5 file

    Returns:
    None
    """

    with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2, h5py.File(output_file, 'w') as out_f:
        datasets_key1 = list(f1.keys())  
        datasets_key2 = list(f2.keys())
        #assert datasets_key1 == datasets_key2, f"The datasets in the two files are not the same. \
        #    key1: {datasets_key1}, key2: {datasets_key2}"

        for dset_name in datasets_key1:
            data1 = f1[dset_name][:] # read all data from the dataset
            data2 = f2[dset_name][:]


            if dset_name != 'language':
                combined_data = np.concatenate((data1, data2), axis=0)
                
                print(f"Data1 shape: {data1.shape}")
                print(f"Data2 shape: {data2.shape}")
                print(f"Combined data shape: {combined_data.shape}")
                
                n = append_or_create_dataset(out_f, dset_name, combined_data)
                print(f"Dataset {dset_name} has {n} samples")

            else:
                combined_data = np.concatenate((data1, data2))
                print("language")
                print(f"Data1 shape: {len(data1)}")
                print(f"Data2 shape: {len(data2)}")
                print(f"Combined data shape: {len(combined_data)}")
                
                n = append_or_create_dataset(out_f, dset_name, combined_data, dtype=h5py.string_dtype(encoding='ascii'))
                print(f"Dataset {dset_name} has {n} samples")
            # create a dataset in the output file
            
           

file1 = '/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/data_hdf5/extra_dataset_no_aug.hdf5'
file2 = '/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/data_hdf5/exist_dataset_no_aug_all.hdf5'
output_file = '/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/data_hdf5/extra_full_color_obj.hdf5'

merge_datasets(file1, file2, output_file)

# import pdb; pdb.set_trace()

# overwrite_dataset(file, 'image_s1', data=img1)
# overwrite_dataset(file, 'image_s2', data=img2)
# overwrite_dataset(file, 'language', data=lang, dtype=h5py.string_dtype(encoding='ascii'))
# overwrite_dataset(file, 'gt_pick', data=pick)
# overwrite_dataset(file, 'gt_place', data=place)

# file.close()
