import cv2
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode
import logging
import os
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)

from detectron2.data.datasets import load_coco_json

from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data.datasets.builtin import register_all_mimc
from detectron2.data import DatasetCatalog
from detectron2.data.datasets.coco import load_sem_seg, register_coco_instances, register_mimc_instances
from detectron2.data.datasets.builtin_meta import ADE20K_SEM_SEG_CATEGORIES, _get_builtin_metadata
register_coco_instances("TRAIN", {}, "/path/to/your/TRAIN.json", "/path/to/your/TRAIN/images")
register_coco_instances("VAL", {}, "/path/to/your/VAL.json", "/path/to/your/VAL/images")
register_coco_instances("TEST", {}, "/path/to/your/TEST.json", "/path/to/your/TEST/images")

MetadataCatalog.get("TRAIN").set(mask_format="RLE")
MetadataCatalog.get("VAL").set(mask_format="RLE")
MetadataCatalog.get("TEST").set(mask_format="RLE")

args = default_argument_parser().parse_args()

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # cfg.DATASETS.TRAIN = ("MIMC_full_s_train",)  # 确保是字符串或字符串列表
    # cfg.DATASETS.TEST = ("MIMC_full_s_val",)  # 如果有验证集
    print(cfg.DATASETS.TRAIN)
    default_setup(cfg, args)
    
    return cfg

cfg = setup(args)

predictor = DefaultPredictor(cfg)

# 加载图像
image_paths = ["xxx1.jpg", "xxx2.jpg", "xxx3.jpg", ] ## your images list
images = [cv2.imread(img_path) for img_path in image_paths]

# 进行预测
outputs = [predictor(image) for image in images]

# 可视化结果
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
for idx, (image, output) in enumerate(zip(images, outputs)):
    path = os.environ["visual_path"]
    print(path)

    v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=0.8, instance_mode=ColorMode.IMAGE_BW)
    vis_output = v.draw_instance_predictions(output["instances"].to("cpu"))
    vis_image = vis_output.get_image()[:, :, ::-1]
    # cv2.imshow(f"Prediction {idx+1}", vis_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(path + "/" + f"output_{idx+1}.jpg", vis_image)
    
