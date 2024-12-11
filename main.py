print("Starting...")
try:
    import sys
    import os
    import bot
    import requests
    import json
    from colorama import Fore, Back, Style
    import time
    import importlib
    from ultralytics import YOLO
    from schema.settings import Settings
    import shutil
    import torch
except Exception as e:
    import traceback
    traceback.print_exc()
    print("You are missing some dependencies. That means you didn't do setup yet. Go back and do autosetup.py")
    input()
    exit()

def prRed(skk): print(Fore.RED, skk, Style.RESET_ALL)
def prGreen(skk): print(Fore.GREEN + skk + Style.RESET_ALL)
def prYellow(skk): print(Fore.YELLOW + skk + Style.RESET_ALL)
def prBlue(skk): print(Fore.BLUE + skk + Style.RESET_ALL)
def prPurple(skk): print(Fore.MAGENTA, skk + Style.RESET_ALL)
def prCyan(skk): print(Fore.CYAN + skk + Style.RESET_ALL)
def prLightGray(skk): print(Fore.WHITE + skk + Style.RESET_ALL)
def prBlack(skk): print(Fore.BLACK + skk + Style.RESET_ALL)

appdataLocation = os.getenv("LOCALAPPDATA")
appdata = os.getenv("APPDATA")
currentDirectory = os.path.dirname(os.path.realpath(__file__))
os.environ['Path'] += f';{appdataLocation}\\Programs\\Python\\Python311\\Scripts'
os.environ['Path'] += f';{appdataLocation}\\Programs\\Python\\Python311\\'
os.environ['Path'] += ';C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8'
os.environ['Path'] += ';C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin'
os.environ['Path'] += ';C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\libnvvp'
os.environ['Path'] += ';C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\lib'
os.environ['CUDA_PATH'] = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8'
os.environ['CUDA_PATH_V11_8'] = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8'

TEST = False

def main():
    customCode = None
    if len(sys.argv) > 3:
        settingsProfile = sys.argv[1]
        yoloVersion = int(sys.argv[2])
        modelFileName = sys.argv[3]
    else:
        print("That's not how you run this. Tsk, tsk, try again.")
        return
    
    if len(sys.argv) > 4:
        try:
            module_dir = os.path.join(appdata, "ai-aimbot-launcher", "customCode")
            if not os.path.exists(module_dir):
                os.makedirs(module_dir)

            os.environ['Path'] += f';{appdata}\\{sys.argv[4].split(".")[0]}\\'
            code_dir = os.path.join(appdata, "ai-aimbot-launcher", "customCode", sys.argv[4].split('.')[0])

            sys.path.append(module_dir)
            sys.path.append(code_dir)

            customCode = importlib.import_module(sys.argv[4]+".main")
        except Exception as err:
            raise Exception("Failed to import custom code")
        
        prBlue(f"Custom Code: {sys.argv[4]}")

    botVersion = 0

    if yoloVersion not in [5, 8]:
        prRed("Invalid YOLO version. Please use 5 or 8")
        return
    
    versionText = ""
    baseFile, fileExtension = os.path.splitext(modelFileName)
    if fileExtension == ".pt":
        botVersion = 0
        versionText = "Fast (PyTorch)"
    elif fileExtension == ".onnx":
        botVersion = 1
        versionText = "Faster (ONNX)"
    elif fileExtension == ".engine":
        botVersion = 2
        versionText = "Fastest (TensorRT)"
        prYellow("First time exporting a model can take up to 20 minutes. Subsequent runs will be immediate.")
    else:
        prRed("Invalid model file extension. Please use .pt, .onnx, or .engine")
        return
    

    modelPTLocation = os.path.join(appdata, "ai-aimbot-launcher", "models", baseFile+".pt")
    settingsPath = os.path.join(appdata, "ai-aimbot-launcher", "aimbotSettings", f"{settingsProfile.lower()}.json")

    # Load settings
    with open(settingsPath, "r") as f:
        settings = json.load(f)
        settings = Settings(**settings)

    # Check if the file exists
    if not os.path.exists(modelPTLocation):
        raise Exception(f"This model does not exist. {baseFile}.pt")


    if botVersion != 0:
        # Check if the file exists
        modelCustomNoEXT = f"{baseFile}v{yoloVersion}{settings.screenShotHeight}{settings.screenShotWidth}Half"
        modelFileName = f"{modelCustomNoEXT}{fileExtension}"
        modelCustomLocation = os.path.join(appdata, "ai-aimbot-launcher", "models", modelFileName)

        if not os.path.exists(modelCustomLocation):
            prYellow(f"Auto Converting model to {fileExtension[1:]}")
            
            modelPTRenamedLocation = os.path.join(appdata, "ai-aimbot-launcher", "models", f"{modelCustomNoEXT}.pt")
            if not os.path.exists(modelPTRenamedLocation):
                shutil.copy(modelPTLocation, modelPTRenamedLocation)

            # Version 8 Export
            if yoloVersion == 8:
                # Load a model
                model = YOLO(modelPTRenamedLocation)  # load an official model

                # Export the model
                model.export(format=fileExtension[1:], imgsz=(settings.screenShotHeight, settings.screenShotWidth), half=True, device=0)  # export at 640px resolution

            # Version 5 Export
            elif yoloVersion == 5:
                import export
                class Opt:
                    def __init__(self):
                        self.weights = modelPTRenamedLocation
                        self.include = [fileExtension[1:]]
                        self.half = True if torch.cuda.is_available() else False
                        self.imgsz = (settings.screenShotHeight, settings.screenShotWidth)
                        self.device = 0 if torch.cuda.is_available() else 'cpu'

                opt = Opt()
                export.main(opt)

            os.remove(modelPTRenamedLocation)
    else:
        modelFileName = f"{baseFile}.pt"

    prGreen(f"Version: {versionText} YOLOv{yoloVersion} from {modelFileName}")  

    prGreen(f"Settings Profile: {settingsProfile}")
    
    if customCode is not None:
        customCode.main(
            version=botVersion,
            settingsProfile=settingsProfile,
            yoloVersion=yoloVersion,
            modelFileName=modelFileName
        )
    else:
        aimbot = bot.Bot(
            version=botVersion,
            settingsProfile=settingsProfile,
            yoloVersion=yoloVersion,
            modelFileName=modelFileName
        )
        aimbot.run()
        

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        prRed(traceback.format_exc())
        prRed(e)
        prYellow("Ask @Wonder for help in our Discord: https://discord.gg/rootkitorg")
    
    for i in range(3, 0, -1):
        prYellow(f"Bot will close in {i} seconds")
        time.sleep(1)