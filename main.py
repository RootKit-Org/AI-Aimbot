import detectron2
import pyautogui
import gc
import pydirectinput

import numpy as np
import os, json, cv2, random
from PIL import Image
import time
import mss

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
predictor = DefaultPredictor(cfg)

# videoGameWindows = pyautogui.getWindowsWithTitle("New World")
# videoGameWindows = pyautogui.getWindowsWithTitle("Halo Infinite")
videoGameWindows = pyautogui.getWindowsWithTitle("Counter-")
videoGameWindow = videoGameWindows[0]

# Find the Window titled exactly "New World" (typically the actual game)
# for window in videoGameWindows:
#     if window.title == "New World":
#     # if window.title == "Halo Infinite":
#         videoGameWindow = window
#         break

# Select that Window
videoGameWindow.activate()
# mssRegion = {"mon": 1, "top": videoGameWindow.top, "left": videoGameWindow.left, "width": videoGameWindow.width, "height": videoGameWindow.height}
# mssRegion = {"mon": 1, "top": videoGameWindow.top + (round(videoGameWindow.height/64) * 20), "left": videoGameWindow.left + (round(videoGameWindow.width/64) * 8), "width": round(videoGameWindow.width/64) * 48, "height": round(videoGameWindow.height/64) * 16}
# mssRegion = {"mon": 1, "top": videoGameWindow.top + round(videoGameWindow.height/3), "left": videoGameWindow.left + round(videoGameWindow.width/3), "width": round(videoGameWindow.width/3), "height": round(videoGameWindow.height/3)}
# mssRegion = {"mon": 1, "top": videoGameWindow.top, "left": videoGameWindow.left + round(videoGameWindow.width/3), "width": round(videoGameWindow.width/3), "height": videoGameWindow.height}
mssRegion = {"mon": 1, "top": videoGameWindow.top+300, "left": videoGameWindow.left, "width": 1280, "height": 250}
# mssRegion = {"mon": 1, "top": videoGameWindow.top + round(videoGameWindow.height/3), "left": videoGameWindow.left, "width": videoGameWindow.width, "height": round(videoGameWindow.height/3)}
# mssRegion = {"mon": 1, "top": videoGameWindow.top, "left": videoGameWindow.left + round(videoGameWindow.width/3), "width": round(videoGameWindow.width/3), "height": videoGameWindow.height}
count = 0
sTime = time.time()
cWidth = mssRegion["width"] / 2
cHeight = mssRegion["height"] / 2
cMargin = 300
print(mssRegion)


sct = mss.mss()
while True:
    npImg = np.delete(np.array(sct.grab(mssRegion)), 3, axis=2)

    outputs = predictor(npImg)
    # print(time.time()-aTime)



    allCenters = outputs['instances'][outputs['instances'].pred_classes== 0].pred_boxes.get_centers()
    cResults = ((allCenters[::,0] > cWidth - cMargin) & (allCenters[::,0] < cWidth + cMargin)) & \
                ((allCenters[::,1] > cHeight - cMargin) & (allCenters[::,1] < cHeight + cMargin))

    allCenters = allCenters.to("cpu")
    targets = np.array(allCenters[cResults])

    # print(len(target))
    if len(targets) > 0:
        # print(target)
        asdf = targets[0] - [cWidth, cHeight+40]
        # print(asdf)

        pydirectinput.move(round(asdf[0]*1), round(asdf[1]*1), relative=True)

    count += 1

    if (time.time() - sTime) > 1:
        # print(time.time()-sTime)
        print(count)
        count = 0
        sTime = time.time()
        gc.collect(generation=0)
    # v = Visualizer(npImg[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #
    # cv2.imshow('sample image',out.get_image()[:, :, ::-1])
    #
    # if (cv2.waitKey(1) & 0xFF) == ord('q'):
    #     exit()
