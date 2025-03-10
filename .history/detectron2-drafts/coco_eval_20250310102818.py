import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import logging
import os
import argparse

# 配置日志
# log_file = f"evaluation.log"  # 日志文件路径
# logging.basicConfig(filename=log_file, level=logging.INFO, 
#                     format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description="测试")
parser.add_argument("--annFile", type=str, help="真实json文件路径")
parser.add_argument("--resFile", type=str, help="预测json文件路径")

args = parser.parse_args()

# 加载标注文件和预测结果文件-json格式
annFile = args.annFile
resFile = args.resFile



# 加载 COCO 数据集
cocoGt = COCO(annFile)  # 加载标注文件
with open(resFile, 'r') as f:
    results = json.load(f)  # 加载预测结果

cocoDt = cocoGt.loadRes(results)

cocoEval = COCOeval(cocoGt, cocoDt, 'segm')  # 使用边界框评估

# print(f"###normal test###")
# print(f"annFile: {annFile}")
# print(f"resFile: {resFile}")
# cocoEval.evaluate()
# cocoEval.accumulate()
print(f"###HAP test###")
print(f"annFile: {annFile}")
print(f"resFile: {resFile}")
cocoEval.hierarchical_evaluate()
cocoEval.hierarchical_accumulate()

# 打印评估结果到日志
cocoEval.summarize()


cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')  # 使用边界框评估

# print(f"###normal test###")
# print(f"annFile: {annFile}")
# print(f"resFile: {resFile}")
# cocoEval.evaluate()
# cocoEval.accumulate()
print(f"###HAP test###")
print(f"annFile: {annFile}")
print(f"resFile: {resFile}")
cocoEval.hierarchical_evaluate()
cocoEval.hierarchical_accumulate()

# 打印评估结果到日志
cocoEval.summarize()

