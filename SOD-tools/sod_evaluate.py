import glob
import os.path
import os
import numpy as np
from PIL import Image
from sod_metrics import Fmeasure,Smeasure,Emeasure
import logging 
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, filename="sod_evaluate.log", format='%(asctime)s - %(levelname)s - %(message)s')

fmesure=Fmeasure() ## FMS
smesure=Smeasure() ## SMS
emesure=Emeasure() ## EMS

maes=[]

pred_folder = os.environ['pred_folder']
gt_folder = os.environ['gt_folder']

pred_files = os.listdir(pred_folder)
gt_files = os.listdir(gt_folder)

gt_base_names = {os.path.splitext(f)[0] for f in gt_files}

for pred_file in tqdm(pred_files):
    
    pred_base_name = os.path.splitext(pred_file)[0]
    pred_path = os.path.join(pred_folder, pred_file)
    gt_path = os.path.join(gt_folder, pred_base_name + '.jpg')
    
    try:
        pred = np.array(Image.open(pred_path).convert("L"))*1.0
        gt=np.array(Image.open(gt_path).convert("L"))*1.0
        
        normalized_pred=pred/255.0
        normalized_gt=gt/255.0
        
        error=np.float32(normalized_pred)-np.float32(normalized_gt)
        
        mae=np.mean(np.abs(error))
        maes.append(mae)

        
        fmesure.step(pred,gt)
        fmesure.get_results()

        
        smesure.step(pred, gt)
        smesure.get_results()

        
        emesure.step(pred, gt)
        emesure.get_results()
    except Exception as e:
        print(f"Error processing file: {pred_path}")
        print(f"Error processing file: {gt_path}")
        print(e)
print("mae")
print(np.average(maes))
print("ems")
print(np.average(emesure.adaptive_ems))
print("fms")
print(np.average(fmesure.adaptive_fms))
print("sms")
print(np.average(smesure.sms))


logging.info(f"{os.environ['pred_folder']}")
logging.info(f"{os.environ['gt_folder']}")
logging.info("mae")
logging.info(np.average(maes))
logging.info("ems")
logging.info(np.average(emesure.adaptive_ems))
logging.info("fms")
logging.info(np.average(fmesure.adaptive_fms))
logging.info("sms")
logging.info(np.average(smesure.sms))