# Useful Tools 

> Codes of this repo are build by [Gintmr](https://gintmr.20250130.xyz/)

## 1. Introduction

This repository contains a set of tools for converting datasets to COCO format. The tools are written in Python and can be used to convert datasets in various formats to COCO format, which is a widely used format for object detection and segmentation tasks.

## 2. Tools

The following tools are included in this repository:


- `merge_multi_datasets.py`: 
  
    > This script merges multiple datasets into a single dataset in COCO format. It takes as input a list of dataset directories and merges them into a single dataset, with each dataset having its own category and instance IDs.

    Usage:将待合并的多个数据集，按照json文件夹与images文件夹的形式分别添加在列表中，然后运行脚本即可

    本脚本在合并时，会同时生成后缀为_categories.json的文件，用于存储合并后数据集的类别信息（本脚本不包含合并测试与验证集的关系，因此保留一个categories副本，便于修改）

- `combine_some_jsonls.py`: 

    > This script combines multiple JSONL files into a single JSONL file. It takes as input a list of JSONL files and combines them into a single JSONL file.

- `get_img_list.py`:

    > This script generates a list of image paths from a directory. It takes as input a directory path and generates a list of image paths in the directory.

- `generate_croods.py`:

    > This script generates random coordinates for a given image's mask. It takes json file as input and generates random coordinates for each mask in the json file.

- Visualize 文件夹:

    > This folder contains a set of scripts for visualizing in many situations. 

    Usage:参见文件夹内README文件

- SOD-tools 文件夹:

    > This folder contains a set of scripts for SOD.

    Usage:参见文件夹内README文件