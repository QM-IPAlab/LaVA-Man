"""
Save existing dataset to hdf5 format
"""

import h5py
import os

file = h5py.File(os.path.join('/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/data_hdf5',
                              'exist_dataset_no_aug.hdf5'), 'r+')

img1 = file['image_s1'][:13387]
img2 = file['image_s2'][:13387]
lang = file['language'][:13387]
pick = file['gt_pick'][:13387]
place = file['gt_place'][:13387]


def overwrite_dataset(f, name, data, dtype=None):
    if name in f:
        del f[name]
    if dtype is not None:
        f.create_dataset(name, data=data, dtype=dtype)
    else:
        f.create_dataset(name, data=data)

import pdb; pdb.set_trace()

overwrite_dataset(file, 'image_s1', data=img1)
overwrite_dataset(file, 'image_s2', data=img2)
overwrite_dataset(file, 'language', data=lang, dtype=h5py.string_dtype(encoding='ascii'))
overwrite_dataset(file, 'gt_pick', data=pick)
overwrite_dataset(file, 'gt_place', data=place)

file.close()
