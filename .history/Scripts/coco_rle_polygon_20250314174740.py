
import os
import json
from pycocotools import mask as maskUtils
import numpy as np
import cv2
from tqdm import tqdm


json_paths = [
    '/Your/Folder/test-taxonomic_cleaned.json',
    '/Your/Folder/train-taxonomic_cleaned.json',
    '/Your/Folder/unseen-taxonomic_cleaned.json',
    '/Your/Folder/val-taxonomic_cleaned.json'
]

for json_path in json_paths:
    output_path = json_path.replace('.json', '_polygon.json')
    out_data = {}
    with open(json_path, 'r') as f:
        data = json.load(f)
        out_data['images'] = data['images']
        out_data['categories'] = data['categories']
        out_data['annotations'] = data['annotations']

        for annotation in tqdm(data['annotations']):
            try:
                rle = annotation['segmentation']
                mask = maskUtils.decode(rle)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                polygons = []
                for contour in contours:
                    contour = contour.flatten().tolist() 
                    if len(contour) > 4: 
                        polygons.append(contour)
                
                if polygons:
                    annotation['segmentation'] = polygons
                    out_data['annotations'].append(annotation)
            except Exception as e:
                print(f"Error processing annotation {annotation['id']}: {e}")
            
        with open(output_path, 'w') as f:
            json.dump(out_data, f)
    
    print(f"{output_path} done")