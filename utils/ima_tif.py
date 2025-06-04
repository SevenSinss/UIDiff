from PIL import Image
import os
from glob import glob
from collections import defaultdict
import numpy as np

def process_tif_files(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    tif_files = glob(os.path.join(input_folder, "*.tif"))
    data_dict = defaultdict(lambda: defaultdict(list))

    for tif_file in tif_files:
        filename = os.path.basename(tif_file)
        parts = filename.split('_')
        if len(parts) < 4:
            print(f"Skipping invalid filename: {filename}")
            continue
        epoch = parts[0]  # e.g., '150000'
        patient_id = parts[-2]  # e.g., 'P001'
        slice_index = int(parts[-1].split('.')[0])  # e.g., '001'
        # print(f"Processing: {filename}, patient_id={patient_id}, slice_index={slice_index}")

        img = Image.open(tif_file)
        img_array = np.array(img)
        data_dict[epoch][patient_id].append((slice_index, img_array))

    for epoch, patient_data in data_dict.items():
        epoch_folder = os.path.join(output_folder, f"epoch_{epoch}")
        os.makedirs(epoch_folder, exist_ok=True)
        for patient_id, slices in patient_data.items():
            slices_sorted = sorted(slices, key=lambda x: x[0])
            img_stack = [Image.fromarray(slice_data, mode='I;16') for _, slice_data in slices_sorted]
            tif_filename = os.path.join(epoch_folder, f"{patient_id}_pred.tif")
            img_stack[0].save(tif_filename, save_all=True, append_images=img_stack[1:], compression='tiff_deflate')
            print(f"Saved {tif_filename}")