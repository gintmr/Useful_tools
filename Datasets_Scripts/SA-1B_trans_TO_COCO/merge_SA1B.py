#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build by XinruiWu
Modified to support command-line arguments

This script is used to merge the SA1B data from different jsons to COCO format.
Usage:
    python merge_sa1b.py --input_dir /path/to/jsons --output /path/to/output.json
"""

import os
import json
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Merge SA1B JSON files into COCO format')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing SA1B JSON files')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file path')
    return parser.parse_args()

def merge_sa1b_to_coco(input_dir, output_path):
    """
    Merge multiple SA1B JSON files into a single COCO format file
    
    Args:
        input_dir: Directory containing SA1B JSON files
        output_path: Path to save the merged COCO format JSON
    """
    # Get all JSON files in the input directory
    json_files = [os.path.join(input_dir, f) 
                  for f in os.listdir(input_dir) 
                  if f.endswith('.json')]
    
    if not json_files:
        raise ValueError(f"No JSON files found in {input_dir}")

    annotations = []
    images = []
    image_id_set = set()  # To track unique image IDs
    
    # Process each JSON file
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
                # Process image data
                if 'image' not in data:
                    print(f"Warning: Missing 'image' key in {json_file}")
                    continue

                image_data = data['image']

                # Ensure unique image IDs
                if 'image_id' in image_data:
                    image_data['id'] = image_data['image_id']
                    del image_data['image_id']
                
                if image_data['id'] in image_id_set:
                    print(f"Warning: Duplicate image ID {image_data['id']} in {json_file}")
                    continue

                image_id_set.add(image_data['id'])
                images.append(image_data)
                
                # Process annotations
                if 'annotations' not in data:
                    print(f"Warning: No annotations in {json_file}")
                    continue
                
                for annotation in data['annotations']:
                    temp_annotation = {
                        'id': annotation['id'],
                        'image_id': image_data['id'],
                        'bbox': annotation['bbox'],
                        'segmentation': annotation['segmentation'],
                        'area': annotation['area'],
                        'iscrowd': 0,
                        'image_name': image_data['file_name']
                    }
                    annotations.append(temp_annotation)

        except json.JSONDecodeError as e:
            print(f"Error decoding {json_file}: {str(e)}")
            continue
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
            continue

    # Create merged data structure
    merged_data = {
        'images': images,
        'annotations': annotations
    }

    # Save the merged data
    with open(output_path, 'w') as f:
        json.dump(merged_data, f, indent=4)

    print(f"\nSuccessfully merged {len(images)} images and {len(annotations)} annotations")
    print(f"Merged data saved to: {output_path}")

if __name__ == "__main__":
    args = parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        raise ValueError(f"Input directory does not exist: {args.input_dir}")

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    merge_sa1b_to_coco(args.input_dir, args.output)