import torch

import pyautogui
import gc

import numpy as np
import os, json, cv2, random
from PIL import Image
import time
import mss
import win32api, win32con

def main():
    # Window title to go after and the height of the screenshots
    videoGameWindowTitle = "Counter"

    screenShotHeight = 500

    # How big the Autoaim box should be around the center of the screen
    aaDetectionBox = 300

    # Autoaim speed
    aaMovementAmp = 2

    # 0 will point center mass, 40 will point around the head in CSGO
    aaAimExtraVertical = 40

    # Set to True if you want to get the visuals
    visuals = False

    # Selecting the correct game window
    try:
        videoGameWindows = pyautogui.getWindowsWithTitle(videoGameWindowTitle)
        videoGameWindow = videoGameWindows[0]
    except:
        print("The game window you are trying to select doesn't exist.")
        print("Check variable videoGameWindowTitle (typically on line 15")
        exit()

    # Select that Window
    videoGameWindow.activate()

    # Setting up the screen shots
    sctArea = {"mon": 1, "top": videoGameWindow.top + round((videoGameWindow.height - screenShotHeight) / 2), "left": videoGameWindow.left, "width": videoGameWindow.width, "height": screenShotHeight}

    #! Uncomment if you want to view the entire screen
    # sctArea = {"mon": 1, "top": 0, "left": 0, "width": 1920, "height": 1080}

    # Starting screenshoting engine
    sct = mss.mss()

    # Calculating the center Autoaim box
    cWidth = sctArea["width"] / 2
    cHeight = sctArea["height"] / 2

    # Used for forcing garbage collection
    count = 0
    sTime = time.time()

    # Loading Yolo5 Small AI Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    # Used for colors drawn on bounding boxes
    COLORS = np.random.uniform(0, 255, size=(1500, 3))

    # Main loop Quit if Q is pressed
    while win32api.GetAsyncKeyState(ord('Q')) == 0:
        # Getting screenshop, making into np.array and dropping alpha dimention.
        npImg = np.delete(np.array(sct.grab(sctArea)), 3, axis=2)

        # Detecting all the objects
        results = model(npImg).pandas().xyxy[0]

        # Filtering out everything that isn't a person
        filteredResults = results[results['class']==0]

        # Returns an array of trues/falses depending if it is in the center Autoaim box or not
        cResults = ((filteredResults["xmin"] > cWidth - aaDetectionBox) & (filteredResults["xmax"] < cWidth + aaDetectionBox)) & \
                    ((filteredResults["ymin"] > cHeight - aaDetectionBox) & (filteredResults["ymax"] < cHeight + aaDetectionBox))

        # Removes persons that aren't in the center bounding box
        targets = filteredResults[cResults]

        # If there are people in the center bounding box
        if len(targets) > 0:
            # All logic is just done on the random person that shows up first in the list
            xMid = round((targets.iloc[0].xmax + targets.iloc[0].xmin) / 2)
            yMid = round((targets.iloc[0].ymax + targets.iloc[0].ymin) / 2)

            mouseMove = [xMid - cWidth, yMid - (cHeight + aaAimExtraVertical)]

            # Moving the mouse
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, round(mouseMove[0] * aaMovementAmp), round(mouseMove[1] * aaMovementAmp), 0, 0) 
        
        # See what the bot sees
        if visuals:
            # Loops over every item identified and draws a bounding box
            for i in range(0, len(results)):
                (startX, startY, endX, endY) = int(results["xmin"][i]), int(results["ymin"][i]), int(results["xmax"][i]), int(results["ymax"][i])

                confidence = results["confidence"][i]

                idx = int(results["class"][i])

                # draw the bounding box and label on the frame
                label = "{}: {:.2f}%".format(results["name"][i], confidence * 100)
                cv2.rectangle(npImg, (startX, startY), (endX, endY),
                    COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(npImg, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        # Forced garbage cleanup every second
        count += 1
        if (time.time() - sTime) > 1:
            print(count)
            count = 0
            sTime = time.time()

            gc.collect(generation=0)

        # See visually what the Aimbot sees
        if visuals:
            cv2.imshow('Live Feed', npImg)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                exit()

if __name__ == "__main__":
    main()