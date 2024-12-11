import torch
import numpy as np
import cv2
import time
import win32api
import win32con
import pandas as pd
import gc
from utils.general import (cv2, non_max_suppression, xyxy2xywh)
import os
import json
from schema.settings import Settings
import pygetwindow
import time
import bettercam
from colorama import Fore, Style, init
from ultralytics import YOLO
from ultralytics.utils import ops
import random
import requests

def prRed(skk): print(Fore.RED + skk + Style.RESET_ALL)
def prGreen(skk): print(Fore.GREEN + skk + Style.RESET_ALL)
def prYellow(skk): print(Fore.YELLOW + skk + Style.RESET_ALL)
def prBlue(skk): print(Fore.BLUE + skk + Style.RESET_ALL)
def prPurple(skk): print(Fore.MAGENTA + skk + Style.RESET_ALL)
def prCyan(skk): print(Fore.CYAN + skk + Style.RESET_ALL)
def prLightGray(skk): print(Fore.WHITE + skk + Style.RESET_ALL)
def prBlack(skk): print(Fore.BLACK + skk + Style.RESET_ALL)

COLORS = np.random.uniform(0, 255, size=(1500, 3))
TEST_MODE = False

class Bot():
    def __init__(
            self,
            version: int,
            settingsProfile: str = "default",
            yoloVersion: int = 5,
            modelFileName: str = "yolov5s.pt",
        ):
        self.settingsProfile = settingsProfile
        self.version = version
        self.yoloVersion = yoloVersion
        self.modelFileName = modelFileName
        self._prepPaths()
        self._loadSettings()

    def _prepPaths(self):
        self.appdataLocation = os.getenv("APPDATA")
        self.settingsPath = os.path.join(self.appdataLocation, "ai-aimbot-launcher", "aimbotSettings", f"{self.settingsProfile.lower()}.json")
        self.gamePath = os.path.join(self.appdataLocation, "ai-aimbot-launcher", "gameSettings", f"{self.settingsProfile.lower()}.json")

    def _loadSettings(self):
        with open(self.settingsPath, "r") as f:
            settings = json.load(f)
            self.settings = Settings(**settings)

    def _gameSelection(self):
        videoGameWindow = None
        # Selecting the correct game window
        if self.settings.gameTitle is not None:
            prYellow(f"Attempting to find game window for {self.settings.gameTitle}")
            videoGameWindows = pygetwindow.getWindowsWithTitle(self.settings.gameTitle)
            if len(videoGameWindows) > 0:
                videoGameWindow = videoGameWindows[0]
                prGreen(f"Selected Game Window: {videoGameWindow.title}")

        if videoGameWindow is None:
            print("Loading window selection...")
            videoGameWindows = pygetwindow.getAllWindows()
            while True:
                try:
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
                        prRed("You didn't enter a valid number. Please try again.")
                        continue
                    # "save" that window as the chosen window for the rest of the script
                    videoGameWindow = videoGameWindows[userInput]
                except Exception as e:
                    prRed("Failed to select game window: {}".format(e))
                    continue  
                break

        # Activate that Window
        activationRetries = 30
        activationSuccess = False
        while (activationRetries > 0):
            try:
                videoGameWindow.activate()
                activationSuccess = True
                break
            except pygetwindow.PyGetWindowException as we:
                prYellow("Failed to activate game window: {}".format(str(we)))
                print("Trying again... (you should switch to the game now)")
            except Exception as e:
                prYellow("Failed to activate game window: {}".format(str(e)))
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
            raise Exception("Failed to activate game window")
        
        prGreen("Successfully activated the game window...")

        # Starting screenshoting engine
        left = ((videoGameWindow.left + videoGameWindow.right) // 2) - (self.settings.screenShotWidth // 2)
        top = videoGameWindow.top + \
            (videoGameWindow.height - self.settings.screenShotHeight) // 2
        right, bottom = left + self.settings.screenShotWidth, top + self.settings.screenShotHeight

        region: tuple = (left, top, right, bottom)

        # Calculating the center Autoaim box
        self.cWidth: int = self.settings.screenShotWidth // 2
        self.cHeight: int = self.settings.screenShotHeight // 2
        self.centerScreen = [self.cWidth, self.cHeight]

        try:
            self.camera = bettercam.create(region=region, output_color="BGRA", max_buffer_len=512)
            if self.camera is None:
                raise Exception("Your Camera Failed!\nAsk @Wonder for help in our Discord in the #bot-ai-aimbot channel ONLY: https://discord.gg/rootkitorg")
        
            self.camera.start(target_fps=120, video_mode=True)
        except Exception as e:
            prRed("Failed to start camera")
            prRed("Common reasons are because the Window minimized, or the Window was not on your Primary monitor.")
            prRed("Ask @Wonder for help in our Discord in the #bot-ai-aimbot channel ONLY: https://discord.gg/rootkitorg")
            raise e
    
    def _loadModel(self):
        if self.version == 0 and self.yoloVersion == 5:
            self._yolov5Torch()
        elif self.version == 1 and self.yoloVersion == 5:
            self._yolov5Onnx()
        elif self.version == 2 and self.yoloVersion == 5:
            self._yolov5Tensorrt()
        elif self.version == 0 and self.yoloVersion == 8:
            self._yolov8Torch()
        elif self.version == 1 and self.yoloVersion == 8:
            self._yolov8Onnx()
        elif self.version == 2 and self.yoloVersion == 8:
            self._yolov8Tensorrt()
        else:
            raise Exception("Invalid model version")

    def _yolov5Torch(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom',
                        path=os.path.join(self.appdataLocation, "ai-aimbot-launcher", "models", self.modelFileName),
        )
        
        if torch.cuda.is_available():
            self.model.half()

    def _yolov5Onnx(self):
        import onnxruntime as ort
        onnxProvider = ""
        if self.settings.onnxChoice == 1:
            onnxProvider = "CPUExecutionProvider"
        elif self.settings.onnxChoice == 2:
            onnxProvider = "DmlExecutionProvider"
        elif self.settings.onnxChoice == 3:
            import cupy as cp
            onnxProvider = "CUDAExecutionProvider"

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        path = os.path.join(self.appdataLocation, "ai-aimbot-launcher", "models", self.modelFileName)

        self.model = ort.InferenceSession(path, sess_options=so, providers=[
                                    onnxProvider])
        
    def _yolov5Tensorrt(self):
        # Loading Yolo5 Small AI Model
        from models.common import DetectMultiBackend
        path = os.path.join(self.appdataLocation, "ai-aimbot-launcher", "models", self.modelFileName
                            )
        self.model = DetectMultiBackend(path, device=torch.device(
            'cuda'), dnn=False, data='', fp16=True)
        self.modelNames = self.model.names

    def _yolov8Torch(self):
        self.model = YOLO(os.path.join(self.appdataLocation, "ai-aimbot-launcher", "models", self.modelFileName))
        self.modelNames = self.model.names

    def _yolov8Onnx(self):
        import onnxruntime as ort
        onnxProvider = ""
        if self.settings.onnxChoice == 1:
            onnxProvider = "CPUExecutionProvider"
        elif self.settings.onnxChoice == 2:
            onnxProvider = "DmlExecutionProvider"
        elif self.settings.onnxChoice == 3:
            import cupy as cp
            onnxProvider = "CUDAExecutionProvider"

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        path = os.path.join(self.appdataLocation, "ai-aimbot-launcher", "models", self.modelFileName)

        self.model = ort.InferenceSession(path, sess_options=so, providers=[
                                    onnxProvider])

    def _yolov8Tensorrt(self):
        from ultralytics.nn import autobackend
        path = os.path.join(self.appdataLocation, "ai-aimbot-launcher", "models", self.modelFileName)

        self.model = autobackend.AutoBackend(
            weights=path,
            fp16=True,
            device=torch.device('cuda'),
        )
        self.modelNames = self.model.names

    def _imageProcessing(self):
        if self.version in [0, 1]:
            return self._generalImageProcessing()
        elif self.version == 2:
            return self._tensorrtImageProcessing()

    def _generalImageProcessing(self):
        # Getting Frame
        npImg = np.array(self.camera.get_latest_frame())

        if self.settings.useMask:
            if self.settings.maskLeft:
                npImg[-self.settings.maskHeight:, :self.settings.maskWidth, :] = 0
            else:
                npImg[-self.settings.maskHeight:, -self.settings.maskWidth:, :] = 0

        # If Nvidia, do this
        if self.settings.onnxChoice == 3:
            # Normalizing Data
            im = torch.from_numpy(npImg).to('cuda')
            if im.shape[2] == 4:
                # If the image has an alpha channel, remove it
                im = im[:, :, :3,]

            im = torch.movedim(im, 2, 0)
            im = im.half()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]
        # If AMD or CPU, do this
        else:

            # Normalizing Data
            if self.version == 0:
                im = torch.from_numpy(npImg)
                if im.shape[2] == 4:
                    # If the image has an alpha channel, remove it
                    im = im[:, :, :3,]

                im = torch.movedim(im, 2, 0)
                im = im.unsqueeze(0)

            else:
                # Normalizing Data
                im = np.array([npImg])
                if im.shape[3] == 4:
                    # If the image has an alpha channel, remove it
                    im = im[:, :, :, :3]
                im = im / 255
                im = im.astype(np.half)
                im = np.moveaxis(im, 3, 1)

        return npImg, im

    def _tensorrtImageProcessing(self):
        import cupy as cp

        npImg = cp.array([self.camera.get_latest_frame()])
        if npImg.shape[3] == 4:
            # If the image has an alpha channel, remove it
            npImg = npImg[:, :, :, :3]

        if self.settings.useMask:
            if self.settings.maskLeft:
                npImg[:, -self.settings.maskHeight:, :self.settings.maskWidth, :] = 0
            else:
                npImg[:, -self.settings.maskHeight:, -self.settings.maskWidth:, :] = 0

        im = npImg / 255
        im = im.astype(cp.half)

        im = cp.moveaxis(im, 3, 1)
        im = torch.from_numpy(cp.asnumpy(im)).to('cuda')

        return npImg, im      

    def _predict(self, image):
        optionsv5 = [
            self._predictTorchYolov5,
            self._predictOnnxYolov5,
            self._predictTensorrtYolov5,
        ]
        optionsv8 = [
            self._predictTorchYolov8,
            self._predictOnnxYolov8,
            self._predictTensorrtYolov5
        ]

        if self.yoloVersion == 5:
            options = optionsv5
        else:
            options = optionsv8

        return options[self.version](image)

    def _predictTorchYolov5(self, image):
        return self.model(image, size=self.settings.screenShotHeight)

    def _predictOnnxYolov5(self, image):
        # If Nvidia, do this
        if self.settings.onnxChoice == 3:
            import cupy as cp
            return torch.from_numpy(self.model.run(None, {'images': cp.asnumpy(image)})[0]).to('cpu')
        # If AMD or CPU, do this
        else:
            return torch.from_numpy(self.model.run(None, {'images': np.array(image)})[0]).to('cpu')

    def _predictTensorrtYolov5(self, image):
        return self.model(image)

    def _predictTorchYolov8(self, image):
        half = False
        if torch.cuda.is_available():
            half = True
        return self.model(
            image,
            half=half,
            imgsz=self.settings.screenShotHeight,
            classes=0,
            conf=self.settings.confidence,
            iou=self.settings.confidence,
            verbose=False
        )

    def _predictOnnxYolov8(self, image):
        # If Nvidia, do this
        if self.settings.onnxChoice == 3:
            import cupy as cp
            return self.model.run(None, {'images': cp.asnumpy(image)})
        # If AMD or CPU, do this
        else:
            return self.model.run(None, {'images': np.array(image)})

    def _targetProcessing(self, outputs, image):
        # Suppressing results that dont meet thresholds
        if self.version in [0, 1, 2] and self.yoloVersion == 5:
            return self._targetProcessingYolov5(outputs, image)
        elif self.version == 0 and self.yoloVersion == 8:
            return self._targetProcessingYolov8Torch(outputs, image)
        elif self.version == 1 and self.yoloVersion == 8:
            return self._targetProcessingYolov8Onnx(outputs, image)
        elif self.version == 2 and self.yoloVersion == 8:
            return self._targetProcessingYolov8Tensorrt(outputs, image)

    def _targetProcessingYolov5(self, outputs, image):
        pred = non_max_suppression(
            outputs,
            conf_thres=self.settings.confidence,
            iou_thres=self.settings.confidence,
            classes=0,
            agnostic=False,
            max_det=1000
        )

        # Converting output to usable cords
        targets = []
        for i, det in enumerate(pred):
            s = ""
            gn = torch.tensor(image.shape)[[0, 0, 0, 0]]
            if len(det):
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {int(c)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    targets.append((xyxy2xywh(torch.tensor(xyxy).view(
                        1, 4)) / gn).view(-1).tolist() + [float(conf)])  # normalized xywh

        return pd.DataFrame(
            targets, columns=['current_mid_x', 'current_mid_y', 'width', "height", "confidence"])

    def _targetProcessingYolov8Torch(self, outputs, image):
        # Converting output to usable cords
        targets = []
        for result in outputs:
            for box in result.boxes:
                # Reshape box.conf to a 2D tensor of size (1, 1)
                box_conf = box.conf.view(1, 1)

                # Concatenate box.xywh and box_conf along dim=1
                box_data_tensor = torch.cat((box.xywh, box_conf), dim=1)

                # Convert the tensor to a numpy array and reshape it to (5,)
                targets.append(box_data_tensor.cpu().numpy().reshape(-1))

        if self.settings.visuals:
            self.annotatedFrame = result.plot()

        return pd.DataFrame(
            targets, columns=['current_mid_x', 'current_mid_y', 'width', "height", "confidence"])

    def _targetProcessingYolov8Onnx(self, outputs, image):
        predictions = np.squeeze(outputs[0]).T

        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > .4, :]
        scores = scores[scores > .4]
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        boxes = predictions[:, :4]
        input_shape = np.array([320, 320, 320, 320])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([320, 320, 320, 320])

        targets = []
        for i, box in enumerate(boxes):
            if class_ids[i] == 0:
                targets.append(np.concatenate((box, scores[i]), axis=None))

        return pd.DataFrame(
            targets, columns=['current_mid_x', 'current_mid_y', 'width', "height", "confidence"])

    def _targetProcessingYolov8Tensorrt(self, outputs, image):
        pred = ops.non_max_suppression(
            outputs,
            conf_thres=self.settings.confidence,
            iou_thres=self.settings.confidence,
            classes=0,
            agnostic=False,
            max_det=1000
        )

        targets = []
        for i, det in enumerate(pred):
            s = ""
            gn = torch.tensor(image.shape)[[0, 0, 0, 0]]
            if len(det):
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.modelNames[int(c)]}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    targets.append((ops.xyxy2xywh(torch.tensor(xyxy).view(
                        1, 4)) / gn).view(-1).tolist() + [float(conf)])  # normalized xywh

        return pd.DataFrame(
            targets, columns=['current_mid_x', 'current_mid_y', 'width', "height", "confidence"])

    def _aimingLogic(self, targets, lastMidCoord=None):
        # If there are people in the center bounding box
        if len(targets) > 0:
            if (self.settings.centerOfScreen):
                # Compute the distance from the center
                targets["dist_from_center"] = np.sqrt((targets.current_mid_x - self.centerScreen[0])**2 + (targets.current_mid_y - self.centerScreen[1])**2)

                # Sort the data frame by distance from center
                targets = targets.sort_values("dist_from_center")

            # Get the last persons mid coordinate if it exists
            if lastMidCoord:
                targets['last_mid_x'] = lastMidCoord[0]
                targets['last_mid_y'] = lastMidCoord[1]
                # Take distance between current person mid coordinate and last person mid coordinate
                targets['dist'] = np.linalg.norm(
                    targets.iloc[:, [0, 1]].values - targets.iloc[:, [4, 5]], axis=1)
                targets.sort_values(by="dist", ascending=False)

            # Take the first person that shows up in the dataframe (Recall that we sort based on Euclidean distance)
            xMid = targets.iloc[0].current_mid_x
            yMid = targets.iloc[0].current_mid_y

            box_height = targets.iloc[0].height
            if self.settings.headshotMode:
                headshot_offset = box_height * self.settings.headshotDistanceModifier
            else:
                headshot_offset = box_height * 0.2

            mouseMove = [xMid - self.cWidth, (yMid - headshot_offset) - self.cHeight]
            mouseMoveX = int(mouseMove[0] * self.settings.movementAmp)
            mouseMoveY = int(mouseMove[1] * self.settings.movementAmp)

            if self.settings.aimShakey:
                mouseMoveX += random.randint(-self.settings.aimShakeyStrength, self.settings.aimShakeyStrength)
                mouseMoveY += random.randint(-self.settings.aimShakeyStrength, self.settings.aimShakeyStrength)

            if self.settings.autoFire and (mouseMoveX + mouseMoveY) <= self.settings.autoFireActivationDistance:
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                time.sleep(0.01)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

            # toggleableModifer = 0x0000
            # if self.settings.toggleable:
            #     toggleableModifer = 0x8000

            if self.settings.fovCircle:
                dist_from_center = np.sqrt(mouseMove[0]**2 + mouseMove[1]**2)

                if dist_from_center > (self.settings.fovCircleRadius * self.settings.fovCircleRadiusDetectionModifier):
                    return [xMid, yMid]

                if dist_from_center <= (self.settings.fovCircleRadius * self.settings.fovCircleRadiusDetectionModifier):
                    # Moving the mouse
                    if self.settings.toggleable:
                        if win32api.GetKeyState(self.settings.activationKey) & 0x8000:
                            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,
                                mouseMoveX, mouseMoveY, 0, 0)
                    else:
                        if win32api.GetKeyState(self.settings.activationKey):
                            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,
                                mouseMoveX, mouseMoveY, 0, 0)
            else:
                if self.settings.toggleable:
                    if win32api.GetKeyState(self.settings.activationKey) & 0x8000:
                        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,
                            mouseMoveX, mouseMoveY, 0, 0)
                else:
                    if win32api.GetKeyState(self.settings.activationKey):
                        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,
                            mouseMoveX, mouseMoveY, 0, 0) 
                    
            return [xMid, yMid]

        else:
            return None

    def _displayVisuals(self, targets, image):
        if self.version == 0 and self.yoloVersion == 8:
            shape = self.annotatedFrame.shape

            center = int(shape[0]/2)

            if self.settings.fovCircle:
                cv2.circle(self.annotatedFrame, (center, center), self.settings.fovCircleRadius, (0, 0, 255), 2)
            
            cv2.imshow("YOLOv8 Inference", self.annotatedFrame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == self.settings.quitKey:
                exit()
            return
        if self.version == 2:
            import cupy as cp
            image = cp.asnumpy(image[0])

        if self.settings.fovCircle:
            center = int(image.shape[0]/2)
            cv2.circle(image, (center, center), self.settings.fovCircleRadius, (0, 0, 255), 2)

        for i in range(0, len(targets)):
            halfW = round(targets["width"][i] / 2)
            halfH = round(targets["height"][i] / 2)
            midX = targets['current_mid_x'][i]
            midY = targets['current_mid_y'][i]
            (startX, startY, endX, endY) = int(
                midX + halfW), int(midY + halfH), int(midX - halfW), int(midY - halfH)

            idx = 0

            # draw the bounding box and label on the frame
            label = "{}: {:.2f}%".format(
                "Human", targets["confidence"][i] * 100)
            cv2.rectangle(image, (startX, startY), (endX, endY),
                            COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            
        cv2.imshow('Live Feed', image)
        if (cv2.waitKey(1) & 0xFF) == self.settings.quitKey:
            exit()

    def run(self):
        self._gameSelection()

        cpsCount = 0
        sTime = time.time()

        self._loadModel()

        lastMidCoord = None
        while win32api.GetAsyncKeyState(self.settings.quitKey) == 0:
            originalImage, editedImage = self._imageProcessing()

            outputs = self._predict(editedImage)

            targets = self._targetProcessing(outputs, editedImage)

            lastMidCoord = self._aimingLogic(targets, lastMidCoord)

            if self.settings.visuals:
                self._displayVisuals(targets, originalImage)

            if self.settings.displayCPS:
                cpsCount += 1
                if (time.time() - sTime) > 1:
                    cpsString = "CPS: {}".format(cpsCount)
                    if cpsCount > 100:
                        prBlue(cpsString)
                    elif cpsCount > 60:
                        prGreen(cpsString)
                    elif cpsCount > 40:
                        prYellow(cpsString)
                    else:
                        prRed(cpsString)
                    cpsCount = 0
                    sTime = time.time()
            
        self.camera.stop()

if __name__ == "__main__":
    try:
        bot = Bot(5, "111")
        bot.run()
    except Exception as e:
        import traceback
        print(e)
        traceback.print_exc()
        input("Press Enter to continue")