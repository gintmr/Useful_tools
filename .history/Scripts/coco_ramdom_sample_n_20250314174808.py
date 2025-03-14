import json
import random 
from tqdm import tqdm

def random_sample(anno_path, num_samples=50):
    with open (anno_path, 'r') as f:
        data = json.load(f)
        kimage_data = data['images']
        random.shuffle(kimage_data)
        sample_kimage_data = kimage_data[:num_samples]
        sampled_data['images'] = sample_kimage_data
        sampled_data['categories'] = data['categories']
        for kimage in tqdm(sample_kimage_data, desc="Sampling annotations"):
            image_id = kimage['id']
            for kannotation in data['annotations']:
                if kannotation['image_id'] == image_id:
                    sampled_data['annotations'].append(kannotation)
    return sampled_data

sampled_data = {
    "images": [],
    "categories": [],
    "annotations": [],
}

anno_path = '/path-to-your-datatsets.json'
num_samples = 10
sampled_anno_path = anno_path.replace('.json', f'_sampled_{num_samples}.json')
sampled_data = random_sample(anno_path, num_samples=num_samples)

with open(sampled_anno_path, 'w') as f:
    json.dump(sampled_data, f)
    print('Randomly sampled {} images and their annotations from {}.'.format(len(sampled_data['images']), anno_path))
