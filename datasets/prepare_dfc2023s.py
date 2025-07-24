#!/usr/bin/env python3
"""
Utility script to prepare DFC2023S dataset for HTC-DC Net.
This script:
1. Analyzes the DFC2023S dataset structure
2. Creates train/val/test split files in the format expected by HTC-DC Net
3. Generates a data split directory structure compatible with the original code
"""

import os
import glob
import random
import argparse
import numpy as np
from tqdm import tqdm
import shutil

def parse_arguments():
    parser = argparse.ArgumentParser(description='Prepare DFC2023S dataset for HTC-DC Net')
    parser.add_argument('--dfc2023s_dir', type=str, default='/home/asfand/Ahmad/datasets/DFC2023S/',
                        help='Path to DFC2023S dataset directory')
    parser.add_argument('--mini_dfc2023s_dir', type=str, default='/home/asfand/Ahmad/datasets/DFC2023Amini/',
                        help='Path to minimized DFC2023S dataset directory for debugging')
    parser.add_argument('--output_dir', type=str, default='/home/asfand/Ahmad/HTC-DC-Net-main/data/DFC2023S/',
                        help='Output directory for HTC-DC Net compatible split files')
    parser.add_argument('--mini', action='store_true', help='Use minimized dataset for debugging')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--validation_ratio', type=float, default=0.0, 
                        help='Ratio of training data to use for validation (0 to use the existing validation split)')
    parser.add_argument('--symbolic_links', action='store_true', help='Create symbolic links for dataset files')
    return parser.parse_args()

def create_splits(args):
    """Create train, validation, and test split files."""
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which dataset directory to use
    data_dir = args.mini_dfc2023s_dir if args.mini else args.dfc2023s_dir
    print(f"Using dataset from: {data_dir}")
    
    # Get dataset files
    train_images = sorted(glob.glob(os.path.join(data_dir, 'train', 'rgb', '*.tif')))
    valid_images = sorted(glob.glob(os.path.join(data_dir, 'valid', 'rgb', '*.tif')))
    test_images = sorted(glob.glob(os.path.join(data_dir, 'test', 'rgb', '*.tif')))
    
    print(f"Found {len(train_images)} training images, {len(valid_images)} validation images, and {len(test_images)} test images.")
    
    # Extract base filenames without extension
    train_files = [os.path.splitext(os.path.basename(f))[0] for f in train_images]
    valid_files = [os.path.splitext(os.path.basename(f))[0] for f in valid_images]
    test_files = [os.path.splitext(os.path.basename(f))[0] for f in test_images]
    
    # Create additional validation split from training data if requested
    if args.validation_ratio > 0:
        # Shuffle the training files
        random.shuffle(train_files)
        
        # Calculate split indices
        val_size = int(len(train_files) * args.validation_ratio)
        
        # Split the data
        custom_val_files = train_files[:val_size]
        custom_train_files = train_files[val_size:]
        
        # Save custom train/val splits
        with open(os.path.join(args.output_dir, 'train.txt'), 'w') as f:
            f.write('\n'.join(custom_train_files))
        
        with open(os.path.join(args.output_dir, 'val.txt'), 'w') as f:
            f.write('\n'.join(custom_val_files))
            
        print(f"Created custom splits: {len(custom_train_files)} training and {len(custom_val_files)} validation files.")
    else:
        # Use the provided train/val splits
        with open(os.path.join(args.output_dir, 'train.txt'), 'w') as f:
            f.write('\n'.join(train_files))
        
        with open(os.path.join(args.output_dir, 'val.txt'), 'w') as f:
            f.write('\n'.join(valid_files))
            
        print(f"Using original splits: {len(train_files)} training and {len(valid_files)} validation files.")
    
    # Save test split
    with open(os.path.join(args.output_dir, 'test.txt'), 'w') as f:
        f.write('\n'.join(test_files))
    
    print(f"Created test split with {len(test_files)} files.")
    return data_dir

def create_symbolic_links(args, data_dir):
    """Create symbolic links to organize the data in the format expected by HTC-DC Net."""
    
    target_dir = os.path.join(args.output_dir, 'data')
    os.makedirs(os.path.join(target_dir, 'image'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'ndsm'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'mask'), exist_ok=True)
    
    print("Creating symbolic links for dataset files...")
    
    # Process all splits
    for split in ['train', 'valid', 'test']:
        rgb_files = glob.glob(os.path.join(data_dir, split, 'rgb', '*.tif'))
        
        for rgb_path in tqdm(rgb_files, desc=f"Processing {split} split"):
            filename = os.path.basename(rgb_path)
            base_name = os.path.splitext(filename)[0]
            
            # Create symbolic links with expected suffixes
            # RGB image -> _IMG.tif
            img_target = os.path.join(target_dir, 'image', f"{base_name}_IMG.tif")
            if not os.path.exists(img_target):
                os.symlink(rgb_path, img_target)
            
            # DSM -> _AGL.tif (Above Ground Level = nDSM)
            dsm_path = os.path.join(data_dir, split, 'dsm', f"{base_name}.tif")
            if os.path.exists(dsm_path):
                dsm_target = os.path.join(target_dir, 'ndsm', f"{base_name}_AGL.tif")
                if not os.path.exists(dsm_target):
                    os.symlink(dsm_path, dsm_target)
            
            # Semantic mask -> _BLG.tif (Building mask)
            sem_path = os.path.join(data_dir, split, 'sem', f"{base_name}.tif")
            if os.path.exists(sem_path):
                mask_target = os.path.join(target_dir, 'mask', f"{base_name}_BLG.tif")
                if not os.path.exists(mask_target):
                    os.symlink(sem_path, mask_target)
    
    print(f"Dataset prepared successfully in {args.output_dir}")

def main():
    args = parse_arguments()
    print("Preparing DFC2023S dataset for HTC-DC Net...")
    
    data_dir = create_splits(args)
    
    if args.symbolic_links:
        create_symbolic_links(args, data_dir)
    else:
        print("Skipping symbolic links creation. Split files created successfully.")

if __name__ == '__main__':
    main()