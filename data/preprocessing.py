"""Image pre-processing utils"""

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import Window
import os
import shutil
import random

def process_sentinel(input_folder, output_path):
    """
    Process Sentinel-2 JP2 files: reproject, resample, scale, and save as a GeoTIFF.
    """
    # Identify band files
    bands = ['B02', 'B03', 'B04', 'B08']
    band_paths = {}
    for f in os.listdir(input_folder):
        for band in bands:
            if f.endswith(f'_{band}_10m.jp2'):
                band_paths[band] = os.path.join(input_folder, f)
                break
    if len(band_paths) != 4:
        raise ValueError("Missing one or more Sentinel-2 band files.")

    # Determine target transform from the first band
    with rasterio.open(band_paths['B02']) as src:
        dst_crs = 'EPSG:4674'
        target_res = 0.000269  # degrees per pixel
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds,
            resolution=(target_res, target_res))
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'count': 1,
            'dtype': 'float32'
        })

    # Process each band and collect data
    sentinel_data = []
    for band in bands:
        with rasterio.open(band_paths[band]) as src:
            data = np.zeros((height, width), dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear)
            sentinel_data.append(data)

    # Stack bands and scale to 0-255
    sentinel_stack = np.stack(sentinel_data, axis=0)
    scaled_stack = []
    for i in range(sentinel_stack.shape[0]):
        band_data = sentinel_stack[i]
        min_val = np.nanmin(band_data)
        max_val = np.nanmax(band_data)
        if max_val == min_val:
            scaled = np.zeros_like(band_data, dtype=np.uint8)
        else:
            scaled = ((band_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        scaled_stack.append(scaled)
    scaled_stack = np.stack(scaled_stack, axis=0)
    
    # Save processed Sentinel-2
    with rasterio.open(output_path, 'w',
                       driver='GTiff',
                       width=width,
                       height=height,
                       count=4,
                       dtype='uint16',
                       crs=dst_crs,
                       transform=transform) as dst:
        dst.write(scaled_stack)
        # Extract acquisition date from filename
        date_str = os.path.basename(band_paths['B02']).split('_')[1]
        dst.update_tags(acquisition_date=date_str)

def process_sentinel_fast(input_folder, output_path):
    """
    Efficiently read Sentinel-2 JP2 files (B02, B03, B04, B08) and save as a single GeoTIFF.
    """
    # Bands of interest
    bands = ['B02', 'B03', 'B04', 'B08']
    band_files = {band: None for band in bands}

    # Locate band files
    for f in os.listdir(input_folder):
        for band in bands:
            if f.endswith(f'_{band}_10m.jp2'):
                band_files[band] = os.path.join(input_folder, f)
                break

    if any(v is None for v in band_files.values()):
        raise ValueError("One or more required band files are missing.")

    # Open all bands simultaneously
    datasets = [rasterio.open(band_files[band]) for band in bands]

    # Use metadata from the first band
    ref = datasets[0]
    profile = ref.profile
    profile.update({
        'count': 4,
        'driver': 'GTiff',
        'dtype': ref.dtypes[0]  # usually 'uint16'
    })

    # Read all bands in memory before writing
    data_stack = np.stack([ds.read(1) for ds in datasets], axis=0)

    # Write to a single GeoTIFF
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data_stack)
        date_str = os.path.basename(band_files['B02']).split('_')[1]
        dst.update_tags(acquisition_date=date_str)

    # Close datasets
    for ds in datasets:
        ds.close()

def clip_prodes(prodes_path, sentinel_processed_path, output_path):
    """
    Clip PRODES dataset to match the processed Sentinel-2 data.
    """
    with rasterio.open(sentinel_processed_path) as src_sentinel:
        sentinel_crs = src_sentinel.crs
        sentinel_transform = src_sentinel.transform
        sentinel_height = src_sentinel.height
        sentinel_width = src_sentinel.width

    with rasterio.open(prodes_path) as src_prodes:
        prodes_data = np.zeros((sentinel_height, sentinel_width), dtype=np.int32)
        reproject(
            source=rasterio.band(src_prodes, 1),
            destination=prodes_data,
            src_transform=src_prodes.transform,
            src_crs=src_prodes.crs,
            dst_transform=sentinel_transform,
            dst_crs=sentinel_crs,
            resampling=Resampling.nearest)

        # Save clipped PRODES
        with rasterio.open(output_path, 'w',
                           driver='GTiff',
                           width=sentinel_width,
                           height=sentinel_height,
                           count=1,
                           dtype=prodes_data.dtype,
                           crs=sentinel_crs,
                           transform=sentinel_transform) as dst:
            dst.write(prodes_data, 1)

def tile_datasets(sentinel_path, prodes_path, root_folder, prefix=''):
    """
    Generate 512x512 non-overlapping tiles for Sentinel-2 and PRODES datasets.
    """
    images_dir = os.path.join(root_folder, 'images')
    labels_dir = os.path.join(root_folder, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    with rasterio.open(sentinel_path) as src_sentinel, \
         rasterio.open(prodes_path) as src_prodes:

        width = src_sentinel.width
        height = src_sentinel.height
        tile_size = 512
        tiles_x = width // tile_size
        tiles_y = height // tile_size
        tile_count = 0

        for j in range(tiles_y):
            for i in range(tiles_x):
                x_off = i * tile_size
                y_off = j * tile_size
                window = Window(x_off, y_off, tile_size, tile_size)

                # Read data
                sentinel_tile = src_sentinel.read(window=window)
                prodes_tile = src_prodes.read(window=window)

                # Skip incomplete tiles
                if sentinel_tile.shape[1:] != (tile_size, tile_size) or \
                   prodes_tile.shape[1:] != (tile_size, tile_size):
                    continue

                # Get transform for the tile
                transform = src_sentinel.window_transform(window)

                # Save Sentinel tile
                tile_name = f"{prefix}_{tile_count:03d}.tif"
                sentinel_meta = src_sentinel.meta.copy()
                sentinel_meta.update({
                    'width': tile_size,
                    'height': tile_size,
                    'transform': transform
                })
                with rasterio.open(os.path.join(images_dir, tile_name), 'w', **sentinel_meta) as dst:
                    dst.write(sentinel_tile)

                # Save PRODES tile
                prodes_meta = src_prodes.meta.copy()
                prodes_meta.update({
                    'width': tile_size,
                    'height': tile_size,
                    'transform': transform,
                    'count': 1
                })
                with rasterio.open(os.path.join(labels_dir, tile_name), 'w', **prodes_meta) as dst:
                    dst.write(prodes_tile)

                tile_count += 1

def convert_forest_to_binary(input_tiff, output_tiff, forest_value):
    """
    Converts a single-band classified raster to a binary mask.
    - Pixels matching `forest_value` → White (255) (Forest)
    - All other pixels → Black (0) (Background)
    """
    with rasterio.open(input_tiff) as src:
        img = src.read(1)  # Read first (only) band
        profile = src.profile.copy()  # Copy metadata

        # Create a binary mask
        mask = np.where(img == forest_value, 255, 0).astype(np.uint8)

        # Update metadata for single-band output
        profile.update({
            "count": 1,
            "dtype": "uint8",
        })

        # Save binary mask
        with rasterio.open(output_tiff, "w", **profile) as dst:
            dst.write(mask, 1)

    print(f"Binary mask saved as {output_tiff}")

def split_dataset(base_path, output_path, train_ratio=0.8, val_ratio=0.15, test_ratio=0.05):
    # Validate input ratios
    if not (0 <= train_ratio <= 1 and 0 <= val_ratio <= 1 and 0 <= test_ratio <= 1):
        raise ValueError("Ratios must be between 0 and 1.")
    if not abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6:
        raise ValueError("The sum of train_ratio, val_ratio, and test_ratio must be 1.")

    # Define paths
    images_path = os.path.join(base_path, 'images')
    labels_path = os.path.join(base_path, 'labels')
    train_images_path = os.path.join(output_path, 'Train', 'images')
    train_labels_path = os.path.join(output_path, 'Train', 'labels')
    val_images_path = os.path.join(output_path, 'Validation', 'images')
    val_labels_path = os.path.join(output_path, 'Validation', 'labels')
    test_images_path = os.path.join(output_path, 'Test', 'images')
    test_labels_path = os.path.join(output_path, 'Test', 'labels')

    # Ensure base paths exist
    if not (os.path.exists(images_path)) or not (os.path.exists(labels_path)):
        raise FileNotFoundError(f"The folders 'images' or 'labels' do not exist in {base_path}.")

    # Create output directories
    os.makedirs(train_images_path, exist_ok=True)
    os.makedirs(train_labels_path, exist_ok=True)
    os.makedirs(val_images_path, exist_ok=True)
    os.makedirs(val_labels_path, exist_ok=True)
    os.makedirs(test_images_path, exist_ok=True)
    os.makedirs(test_labels_path, exist_ok=True)

    # Get list of image and label files
    image_files = [f for f in os.listdir(images_path) if f.endswith('.tif') and os.path.isfile(os.path.join(images_path, f))]
    label_files = [f for f in os.listdir(labels_path) if f.endswith('.tif') and os.path.isfile(os.path.join(labels_path, f))]

    # Ensure corresponding image and label files exist
    image_files = [f for f in image_files if f in label_files]
    label_files = [f for f in label_files if f in image_files]

    # Shuffle the files
    random.shuffle(image_files)

    # Calculate split indices
    total_files = len(image_files)
    train_split = int(total_files * train_ratio)
    val_split = int(total_files * (train_ratio + val_ratio))

    # Split files
    train_files = image_files[:train_split]
    val_files = image_files[train_split:val_split]
    test_files = image_files[val_split:]

    # Function to copy files
    def copy_files(files, images_src, labels_src, images_dest, labels_dest):
        for file in files:
            shutil.copy(os.path.join(images_src, file), os.path.join(images_dest, file))
            shutil.copy(os.path.join(labels_src, file), os.path.join(labels_dest, file))

    # Copy files to respective directories
    copy_files(train_files, images_path, labels_path, train_images_path, train_labels_path)
    copy_files(val_files, images_path, labels_path, val_images_path, val_labels_path)
    copy_files(test_files, images_path, labels_path, test_images_path, test_labels_path)

    print(f"Dataset split completed: {len(train_files)} Train, {len(val_files)} Validation, {len(test_files)} Test")