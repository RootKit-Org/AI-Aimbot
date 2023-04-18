import onnxruntime as ort
import numpy as np
import pyautogui
import pygetwindow
import gc
import numpy as np
import cv2
import time
import win32api
import win32con
import pandas as pd
from utils.general import (cv2, non_max_suppression, xyxy2xywh)
import dxcam
import torch
import torch_directml


def main():
    # Portion of screen to be captured (This forms a square/rectangle around the center of screen)
    screenShotHeight = 320
    screenShotWidth = 320

    # For use in games that are 3rd person and character model interferes with the autoaim
    # EXAMPLE: Fortnite and New World
    aaRightShift = 0

    # An alternative to aaRightShift
    # Mark regions of the screen where your own player character is
    # This will often prevent the mouse from drifting to an edge of the screen
    # Format is (minX, minY, maxX, maxY) to form a rectangle
    # Remember, Y coordinates start at the top and move downward (higher Y values = lower on screen)
    skipRegions: list[tuple] = [
        (0, 0, 0, 0)
    ]

    # Autoaim mouse movement amplifier
    aaMovementAmp = .3
    
    # Person Class Confidence
    confidence = 0.40

    # What key to press to quit and shutdown the autoaim
    aaQuitKey = "P"

    # If you want to main slightly upwards towards the head
    headshot_mode = False

    # Displays the Corrections per second in the terminal
    cpsDisplay = True

    # Set to True if you want to get the visuals
    visuals = False

    # Selecting the correct game window
    try:
        videoGameWindows = pygetwindow.getAllWindows()
        print("=== All Windows ===")
        for index, window in enumerate(videoGameWindows):
            # only output the window if it has a meaningful title
            if window.title != "":
                print("[{}]: {}".format(index, window.title))
        # have the user select the window they want
        try:
            userInput = int(input(
                "Please enter the number corresponding to the window you'd like to select: "))
        except ValueError:
            print("You didn't enter a valid number. Please try again.")
            return
        # "save" that window as the chosen window for the rest of the script
        videoGameWindow = videoGameWindows[userInput]
    except Exception as e:
        print("Failed to select game window: {}".format(e))
        return

    # Activate that Window
    activationRetries = 30
    activationSuccess = False
    while (activationRetries > 0):
        try:
            videoGameWindow.activate()
            activationSuccess = True
            break
        except pygetwindow.PyGetWindowException as we:
            print("Failed to activate game window: {}".format(str(we)))
            print("Trying again... (you should switch to the game now)")
        except Exception as e:
            print("Failed to activate game window: {}".format(str(e)))
            print("Read the relevant restrictions here: https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setforegroundwindow")
            activationSuccess = False
            activationRetries = 0
            break
        # wait a little bit before the next try
        time.sleep(3.0)
        activationRetries = activationRetries - 1
    # if we failed to activate the window then we'll be unable to send input to it
    # so just exit the script now
    if activationSuccess == False:
        return
    print("Successfully activated the game window...")

    

    # Starting screenshoting engine
  
    camera = dxcam.create(device_idx=0)
    if camera is None:
        print("""DXCamera failed to initialize. Some common causes are:
        1. You are on a laptop with both an integrated GPU and discrete GPU. Go into Windows Graphic Settings, select python.exe and set it to Power Saving Mode.
         If that doesn't work, then read this: https://github.com/SerpentAI/D3DShot/wiki/Installation-Note:-Laptops
        2. The game is an exclusive full screen game. Set it to windowed mode.""")
        return
    

    # Calculating the center Autoaim box
    cWidth =  screenShotWidth / 2
    cHeight = screenShotHeight / 2

    # Used for forcing garbage collection
    count = 0
    sTime = time.time()

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_mem_pattern = False
    ort_sess = ort.InferenceSession('yolov5s320.onnx', sess_options=so, providers=['DmlExecutionProvider'])
    def clamp(num, min_value, max_value):
       return max(min(num, max_value), min_value)                                

      
    # Main loop Quit if Q is pressed
    last_mid_coord = None
    while win32api.GetAsyncKeyState(ord(aaQuitKey)) == 0:
        # Getting Frame
        mx, my = win32api.GetCursorPos()

        # Capture a 320 x 320 region around the cursor using DXCam
        # Note: you will need to install and set up DXCam properly for this code to work
        cmx = clamp(mx, 180, 1740)
        cmy = clamp(my, 180, 900)
        region = (cmx - 160, cmy - 160, cmx + 160, cmy + 160)
        
        cap = camera.grab(region = region)
        if cap is None:
            continue

        # Normalizing Data
        npImg = np.array([cap]) / 255
        npImg = npImg.astype(np.half)
        npImg = np.moveaxis(npImg, 3, 1)

        # Run ML Inference
        outputs = ort_sess.run(None, {'images': np.array(npImg)})
        im = torch.from_numpy(outputs[0]).to('cpu')
        pred = non_max_suppression(
            im, confidence, confidence, 0, False, max_det=10)

        # Get targets from ML predictions
        targets = []
        for i, det in enumerate(pred):
            s = ""
            gn = torch.tensor(npImg.shape)[[0, 0, 0, 0]]
            if len(det):
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {int(c)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    # normalized xywh
                    detTensorScreenCoords = (xyxy2xywh(torch.tensor(xyxy).view(
                        1, 4)) / gn).view(-1)
                    detScreenCoords = (
                        detTensorScreenCoords.tolist() + [float(conf)])
                    isSkipped = False
                    for skipRegion in skipRegions:
                        # TODO check logic. there are some rare edge cases.
                        # if min and max are both within the min and max of the other, then we are fully within it
                        detectionWithinSkipRegion = ((xyxy[0] >= skipRegion[0] and xyxy[2] <= skipRegion[2])
                                                     and (xyxy[1] >= skipRegion[1] and xyxy[3] <= skipRegion[3]))
                        # if above top edge, to the right of right edge, below bottom edge, or left of left edge, then there can be no intersection
                        detectionIntersectsSkipRegion = not (
                            xyxy[0] > skipRegion[2] or xyxy[2] < skipRegion[0] or xyxy[1] > skipRegion[3] or xyxy[1] < skipRegion[3])
                        if detectionWithinSkipRegion or detectionIntersectsSkipRegion:
                            isSkipped = True
                            break
                    if isSkipped == False:
                        targets.append(detScreenCoords)

        targets = pd.DataFrame(
            targets, columns=['current_mid_x', 'current_mid_y', 'width', "height", "confidence"])

        # If there are people in the center bounding box
        if len(targets) > 0:
            # Get the last persons mid coordinate if it exists
            if last_mid_coord:
                targets['last_mid_x'] = last_mid_coord[0]
                targets['last_mid_y'] = last_mid_coord[1]
                # Take distance between current person mid coordinate and last person mid coordinate
                targets['dist'] = np.linalg.norm(
                    targets.iloc[:, [0, 1]].values - targets.iloc[:, [4, 5]], axis=1)
                # This ensures the person closest to the crosshairs is the one that's targeted
                targets.sort_values(by="dist", ascending=False)

            # Take the first person that shows up in the dataframe (Recall that we sort based on Euclidean distance)
            xMid = targets.iloc[0].current_mid_x 
            yMid = targets.iloc[0].current_mid_y

            box_height = targets.iloc[0].height
            if headshot_mode:
                headshot_offset = box_height * 0.38
            else:
                headshot_offset = 0
            mouseMove = [xMid - cWidth, (yMid - headshot_offset) - cHeight]
          

            # Moving the mouse
            if win32api.GetKeyState(0x14):
                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(
                    mouseMove[0] * aaMovementAmp), int(mouseMove[1] * aaMovementAmp), 0, 0)
            last_mid_coord = [xMid, yMid]

        else:
            last_mid_coord = None

        
        # Forced garbage cleanup every second
        count += 1
        if (time.time() - sTime) > 1:
            if cpsDisplay:
                print("CPS: {}".format(count))
            count = 0
            sTime = time.time()

            # Uncomment if you keep running into memory issues
            # gc.collect(generation=0)

        # See visually what the Aimbot sees
        if visuals:
            cv2.imshow(cv2WindowName, cap)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                cv2.destroyAllWindows()
                exit()
    camera.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("Please read the below message and think about how it could be solved before posting it on discord.")
        traceback.print_exception(e)
        print(str(e))
        print("Please read the above message and think about how it could be solved before posting it on discord.")
    cv2.destroyAllWindows()
