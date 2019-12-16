import numpy as np
import pylidc as pl
import os
import shutil
import matplotlib.pyplot as plt
from glob import glob
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from plotly.graph_objs import *
from statistics import mean
import scipy as sp

MIN_SLICES = 65
load_dicom = True


def extract_lungs():
    output_folder = "./output"
    np_save_folder = "./np-save"

    if not os.path.exists(np_save_folder):
        os.makedirs(np_save_folder)

    scans = pl.query(pl.Scan)

    for scan in scans:
        # Make lung directory for output
        lung_num = scan.patient_id
        lung_output_folder = os.path.join(output_folder, lung_num)

        print("Converting Lung: {}".format(lung_num))

        # Make Output folder
        if os.path.exists(lung_output_folder):
            shutil.rmtree(lung_output_folder)
        os.makedirs(lung_output_folder)

        np_file = os.path.join(np_save_folder, "{}.npy".format(lung_num))
        if load_dicom:
            # Convert scans to numpy arrays
            images = scan.load_all_dicom_images(False)
            slices = get_pixels_hu(images)
            slices, amt_rm_frm_beg = cut_slices(slices)

            create_anotations(lung_output_folder, scan, slices, amt_rm_frm_beg)

            # np.save(np_file, slices)
        else:
            pass
            # slices = np.load(np_file).astype(np.float64)

        masked = [s for s in slices]
        save_scan(lung_output_folder, masked)


def get_pixels_hu(slices):
    # Stack and convert pixels
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    return image


def create_anotations(lung_output_folder, scan, slices, start_slice):
    slice_anns = {x: 0 for x in range(len(slices))}

    annotations = scan.cluster_annotations()
    mean_annotations = []

    # For annotated nodules iterate over the nodes
    for node in annotations:
        node_mean = mean([n_ann.malignancy for n_ann in node])

        # Assign label of 1 to begnin and 2 to malignant
        if node_mean < 3:
            node_label = 1
        else:
            node_label = 2

        # Get the applicable layer and label them
        node_idxs = node[0].contour_slice_indices
        for idx in node_idxs:
            if idx >= start_slice and idx <= start_slice + MIN_SLICES:
                slice_anns[idx - start_slice] = node_label

    # Write Annotations to file
    with open(os.path.join(lung_output_folder, "annotations.txt"), "w+") as f:
        for key, val in slice_anns.items():
            f.write(f"{key}: {val}\n")


def cut_slices(slices):
    if len(slices) < MIN_SLICES:
        return slices
    else:
        middle = slices.shape[0] / 2
        lower = int(middle - (MIN_SLICES / 2))
        higher = int(middle + (MIN_SLICES / 2))
        return slices[lower:higher, :, :], lower


def save_scan(folder, slices):
    for x in range(len(slices)):
        slice_name = f"{x}.png"

        image_save_loc = os.path.join(folder, slice_name)
        plt.imsave(
            image_save_loc, slices[x], cmap=plt.cm.bone,
        )


if __name__ == "__main__":
    extract_lungs()
