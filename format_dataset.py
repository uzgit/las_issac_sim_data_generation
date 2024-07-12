#!/usr/bin/env python3

import os
import argparse
import shutil
import glob

def process_directory(input_dir):
    # Define subdirectories
    train_dir = os.path.join(input_dir, 'train')
    images_dir = os.path.join(train_dir, 'images')
    masks_dir = os.path.join(train_dir, 'masks')
    
    # Create train and subdirectories
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    # Get the directory name without the full path
    dir_name = os.path.basename(os.path.normpath(input_dir))
    
    # Process image files
    for img_file in glob.glob(os.path.join(input_dir, 'image_*.png')):
        new_name = f"{dir_name}_{os.path.basename(img_file)[len('image_'):]}".replace('.png', '') + '.png'
        new_path = os.path.join(images_dir, new_name)
        shutil.move(img_file, new_path)
    
    # Process mask files
    for mask_file in glob.glob(os.path.join(input_dir, 'label_*.png')):
        new_name = f"{dir_name}_{os.path.basename(mask_file)[len('label_'):]}".replace('.png', '') + '.png'
        new_path = os.path.join(masks_dir, new_name)
        shutil.move(mask_file, new_path)
    
    # Remove all other files in the directory
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process image and label files in specified directories.")
    parser.add_argument('directories', type=str, nargs='+', help='Directories to process.')
    
    args = parser.parse_args()
    
    for input_directory in args.directories:
        if not os.path.isdir(input_directory):
            print(f"The specified directory does not exist: {input_directory}")
            continue
        
        process_directory(input_directory)
        print(f"Processing complete for: {input_directory}")

if __name__ == "__main__":
    main()

