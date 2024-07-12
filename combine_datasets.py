#!/usr/bin/env python3

import os
import argparse
import shutil
import glob

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Combine image and label files from multiple directories.")
    parser.add_argument('directories', type=str, nargs='+', help='Directories to copy from.')
    parser.add_argument('--combined_directory', type=str, default='combined_dataset', help='Directory to create for combined files.')
    
    args = parser.parse_args()

    # Create the combined directory
    os.makedirs(args.combined_directory, exist_ok=True)
    os.makedirs(f"{args.combined_directory}/train/images", exist_ok=True)
    os.makedirs(f"{args.combined_directory}/train/masks", exist_ok=True)

    for input_directory in args.directories:
        print(f"processing {input_directory}")

        for file in glob.glob(f"{input_directory}/train/images/*.png"):
            shutil.copy2(file, f"{args.combined_directory}/train/images")
        
        for file in glob.glob(f"{input_directory}/train/masks/*.png"):
            shutil.copy2(file, f"{args.combined_directory}/train/masks")

if __name__ == "__main__":
    main()

