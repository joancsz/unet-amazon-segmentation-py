
import pathlib
import os
from data.preprocessing import (
                                process_sentinel,
                                process_sentinel_fast,
                                clip_prodes,
                                convert_forest_to_binary,
                                tile_datasets,
                                split_dataset
                                )

#Input images
SENTINEL_IMAGES_FOLDER = pathlib.Path("/SENTINEL_IMAGES/")
#Output processed images
SENTINEL_IMAGES_OUTUPUT_FOLDER = pathlib.Path("/SENTINEL_OUTPUT/")

#Prodes image path
PRODES_DATA_PATH = "/prodes_amazonia_legal_2023.tif"

#Processed tiles
TILES_OUTPUT_FOLDER = pathlib.Path("/CUSTOM_TILES/")

#Custom dataset destination
DATASET_FOLDER = pathlib.Path('/CUSTOM_AMAZON/') 

FOREST_VALUE = 100

os.makedirs(TILES_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(SENTINEL_IMAGES_OUTUPUT_FOLDER, exist_ok=True)

for file_folder in list(SENTINEL_IMAGES_FOLDER.glob('*')):
    key = file_folder.name
    output_folder = os.path.join(SENTINEL_IMAGES_OUTUPUT_FOLDER, key)
    os.makedirs(output_folder, exist_ok=True)
    print(key)
    output_sentinel = os.path.join(output_folder, f"{key}.tif")
    output_prodes = os.path.join(output_folder, f"{key}_PRODES.tif")
    output_prodes_binary = os.path.join(output_folder, f"{key}_PRODES_BINARY.tif")
    
    # Process Sentinel-2
    #process_sentinel(file_folder, output_sentinel)
    process_sentinel_fast(file_folder, output_sentinel)

    # Clip PRODES
    clip_prodes(PRODES_DATA_PATH, output_sentinel, output_prodes)
    convert_forest_to_binary(output_prodes, output_prodes_binary, FOREST_VALUE)

    # Generate tiles
    tile_datasets(output_sentinel, output_prodes_binary, TILES_OUTPUT_FOLDER, prefix=key)

split_dataset(TILES_OUTPUT_FOLDER, DATASET_FOLDER)