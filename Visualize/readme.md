> 各文件讲解

- Visualize\cauculate_datasets.py : 统计数据集的token数量指标画出直方图，同时兼具过滤过长数据的功能

- Visualize\visualize-coco-gt.py : 给定需要可视化的图像名，读取coco格式数据集，将GT标注可视化

- Visualize\visulize-sod-mask.py : 给定原始图像与mask二值掩码，将mask覆盖到原图上，实现可视化

- Visualize\preview_jsonl.py : 给定jsonl文件，随机抽取/从头提取/从尾部提取 ~ 数据，输出到新的jsonl文件中