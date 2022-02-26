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

def main():
    # Window title to go after and the height of the screenshots
    videoGameWindowTitle = "Counter-Strike"
    videoGameWindowTitle = "Valorant"
    screenShotHeight = 250

    # How big the Autoaim box should be around the center of the screen
    aaDetectionBox = 300

    # Autoaim speed
    aaMovementAmp = 2

    # 0 will point center mass, 40 will point around the head in CSGO
    aaAimExtraVertical = 40

    # Loading up the object detection model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
    predictor = DefaultPredictor(cfg)

    # Selecting the correct game window
    videoGameWindows = pyautogui.getWindowsWithTitle(videoGameWindowTitle)
    videoGameWindow = videoGameWindows[0]

    # Select that Window
    videoGameWindow.activate()

    # Setting up the screen shots
    sctArea = {"mon": 1, "top": videoGameWindow.top + round((videoGameWindow.height - screenShotHeight) / 2), "left": videoGameWindow.left, "width": videoGameWindow.width, "height": screenShotHeight}
    sct = mss.mss()

    # Calculating the center Autoaim box
    cWidth = sctArea["width"] / 2
    cHeight = sctArea["height"] / 2

    # Used for forcing garbage collection
    count = 0
    sTime = time.time()

    # Main loop
    while True:
        # Getting screenshop, making into np.array and dropping alpha dimention.
        npImg = np.delete(np.array(sct.grab(sctArea)), 3, axis=2)

        # Detecting all the objects
        predictions = predictor(npImg)

        # Removing anything that isn't a human and getting the center of those object boxes
        predCenters = predictions['instances'][predictions['instances'].pred_classes== 0].pred_boxes.get_centers()

        # Returns an array of trues/falses depending if it is in the center Autoaim box or not
        cResults = ((predCenters[::,0] > cWidth - aaDetectionBox) & (predCenters[::,0] < cWidth + aaDetectionBox)) & \
                    ((predCenters[::,1] > cHeight - aaDetectionBox) & (predCenters[::,1] < cHeight + aaDetectionBox))

        # Moves variable from the GPU to CPU
        predCenters = predCenters.to("cpu")

        # Removes all predictions that aren't closest to the center
        targets = np.array(predCenters[cResults])

        # If there are targets in the center box
        if len(targets) > 0:
            # Get the first target
            mouseMove = targets[0] - [cWidth, cHeight + aaAimExtraVertical]

            # Move the mouse
            pydirectinput.move(round(mouseMove[0] * aaMovementAmp), round(mouseMove[1] * aaMovementAmp), relative=True)

        # Forced garbage cleanup every second
        count += 1
        if (time.time() - sTime) > 1:
            count = 0
            sTime = time.time()
            gc.collect(generation=0)

        #! Uncomment to see visually what the Aimbot sees
        # v = Visualizer(npImg[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        # out = v.draw_instance_predictions(predictions["instances"].to("cpu"))
        # cv2.imshow('sample image',out.get_image()[:, :, ::-1])
        # if (cv2.waitKey(1) & 0xFF) == ord('q'):
        #     exit()

if __name__ == "__main__":
    main()