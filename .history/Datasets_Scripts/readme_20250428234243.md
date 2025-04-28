- `merge_multi_datasets.py`: 
  
    > This script merges multiple datasets into a single dataset in COCO format. It takes as input a list of dataset directories and merges them into a single dataset, with each dataset having its own category and instance IDs.

    Usage:将待合并的多个数据集，按照json文件夹与images文件夹的形式分别添加在列表中，然后运行脚本即可

    本脚本在合并时，会同时生成后缀为_categories.json的文件，用于存储合并后数据集的类别信息（本脚本不包含合并测试与验证集的关系，因此保留一个categories副本，便于修改）

- `combine_some_jsonls.py`: 

    > This script combines multiple JSONL files into a single JSONL file. It takes as input a list of JSONL files and combines them into a single JSONL file.

- `get_img_list.py`:

    > This script generates a list of image paths from a directory. It takes as input a directory path and generates a list of image paths in the directory.

- `generate_croods.py`:

    > This script generates random coordinates for a given image's mask. It takes json file as input and generates random coordinates for each mask in the json file

- `replace_specified_key_in_json.py`:

    > This script replaces a specified key in a JSON file with a new value. It takes a JSON file path and a new value as input and replaces the specified key in the JSON file with the new value.

- `coco_rle_polygon.py`: 

    > This script converts COCO RLE format to polygon format. It takes a COCO RLE format mask and converts it to polygon format.

- `coco_ramdom_sample_n.py`:

    > This script randomly samples n images from a COCO dataset. It takes a COCO dataset directory and a number of images to sample as input and generates a new COCO dataset with n randomly sampled images.

- `Datasets_Scripts\SA-1B_trans_TO_COCO`

    > This directory contains the scripts used to convert the SA-1B dataset to COCO format.