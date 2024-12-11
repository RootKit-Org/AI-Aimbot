from colorama import Fore, Style, init
import subprocess
import zipfile
import ctypes
import shutil
import winreg
import glob
import sys
import os

# $nidlNnvB22n9T^4 (idk what this is @elijah, ill just keep it here just incase)
# cmd.exe /c pyarmor gen autoInstall.py "&" pyinstaller -i rootkitLogo.ico --onefile --uac-admin --hidden-import=colorama --hidden-import=subprocess --hidden-import=zipfile --hidden-import=ctypes --hidden-import=shutil --hidden-import=winreg --hidden-import=glob --hidden-import=sys --hidden-import=os dist/autoInstall.py

init() # colorama init

os.system('cls')

# added admin check, admin is needed to automatically add cuda lib to path
def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

if not is_admin():
    print(Fore.RED + "Admin prompt was not accepted, please restart the AutoInstall and accept the admin prompt" + Style.RESET_ALL)
    input("Press Enter to exit")
    sys.exit(1)

# function for extracting the cuDNN and TensorRT zip files
def extract_subfolder(zip_file_path, target_dir):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # List all files in the zip file
        all_files = zip_ref.namelist()
        
        # Find the first subfolder in the zip file
        subfolder = next((f for f in all_files if '/' in f), None)
        
        if subfolder is None:
            return
        
        # Extract each file from the subfolder to the target directory
        for file in all_files:
            if file.startswith(subfolder):
                # Remove the subfolder prefix from the file path
                relative_path = file[len(subfolder):]
                if relative_path and not file.endswith('/'):  # Ensure it's not an empty string and not a directory
                    target_path = os.path.join(target_dir, relative_path)
                    # Create target directory if it doesn't exist
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    # Delete the file if it already exists
                    if os.path.exists(target_path):
                        os.remove(target_path)
                    # Extract the file, overwriting if it exists
                    with zip_ref.open(file) as source, open(target_path, 'wb') as target:
                        shutil.copyfileobj(source, target)

# improved colorama print functions
def pr_red(skk): print(Fore.RED + skk, Style.RESET_ALL)
def pr_light_red(skk): print(Fore.LIGHTRED_EX + skk, Style.RESET_ALL)
def pr_green(skk): print(Fore.GREEN + skk + Style.RESET_ALL)
def pr_light_green(skk): print(Fore.LIGHTGREEN_EX + skk + Style.RESET_ALL)
def pr_blue(skk): print(Fore.BLUE + skk + Style.RESET_ALL)
def pr_light_blue(skk): print(Fore.LIGHTBLUE_EX + skk + Style.RESET_ALL)
def pr_yellow(skk): print(Fore.YELLOW + skk + Style.RESET_ALL)
def pr_light_yellow(skk): print(Fore.LIGHTYELLOW_EX + skk + Style.RESET_ALL)
def pr_magenta(skk): print(Fore.MAGENTA + skk + Style.RESET_ALL)
def pr_light_magenta(skk): print(Fore.LIGHTMAGENTA_EX + skk + Style.RESET_ALL)
def pr_cyan(skk): print(Fore.CYAN + skk + Style.RESET_ALL)
def pr_black(skk): print(Fore.BLACK + skk + Style.RESET_ALL)
def pr_light_black(skk): print(Fore.LIGHTBLACK_EX + skk, Style.RESET_ALL)

# Construct the pattern to match directories
pattern = os.path.join(os.getenv('LOCALAPPDATA'), 'ai_aimbot_launcher', 'app-*', 'resources', 'extras', 'AI_Aimbot')

# Use glob to find directories matching the pattern
matchedPaths = glob.glob(pattern)

# extra path in resources/AI_Aimbot of the rootkit electron app
PATHEXTRA = matchedPaths[0]

def main():
    pr_cyan("Welcome To the AI Aimbot AutoInstall Script - By RootKit")
    pr_green("This script will install all the dependencies needed to run the AI Aimbot")
    input("Press Enter to continue")

    os.system('cls')

    # check for nvidia gpu
    selection = None
    try:
        output = subprocess.check_output('nvidia-smi --query-gpu "name" --format=csv', shell=True, universal_newlines=True)
        lines = output.strip().split('\n')
        if len(lines) >= 2:
            gpuName = lines[1].strip()
        else:
            gpuName = None
    except subprocess.CalledProcessError:
        gpuName = None

    # ask user which version of the AI Aimbot they want to install
    while selection is None:
        pr_green("Please select the version of the AI Aimbot you want to install")
        pr_cyan("If you want a quick install to test the aimbot, select the UNIVERSAL GPU VERSION")
        pr_light_green("If you have an NVIDIA GPU, and want a more advanced faster version, select the correct NVIDIA version")

        pr_yellow("Detected GPU: " + gpuName if gpuName is not None else "No NVIDIA GPU Detected")

        pr_cyan("1. UNIVERSAL GPU VERSION" + (" (Recommended)" if gpuName is None else ""))
        pr_green("2. NVIDIA GTX GPU VERSION" + (" (Recommended)" if "gtx" in gpuName.lower() else ""))
        pr_light_green("3. NVIDIA RTX GPU VERSION" + (" (Recommended)" if "rtx" in gpuName.lower() else ""))

        buffer = input("> ")

        try:
            if any(i == buffer for i in ['1', '2', '3']):
                selection = int(buffer)
            else:
                os.system('cls')
                raise Exception("Please enter a valid number")
        except Exception:
            os.system('cls')
            pr_red("Please enter a valid number")
    
    if selection == 1:
        text = "UNIVERSAL GPU VERSION"
    elif selection == 2:
        text = "NVIDIA GTX GPU VERSION"
    elif selection == 3:
        text = "NVIDIA RTX GPU VERSION"
    pr_green(f"Successfully Selected - {text}")

    input("Press Enter to continue")
    os.system('cls')

    # install python 3.11.9
    pr_cyan("First let's install python 3.11.9, don't worry this won't overwrite your current python installation")
    pr_light_red("JUST CLICK NEXT AND INSTALL")
    pr_yellow("IF you have python 3.11.9 installed already, then you can just close the installer")

    os.system(PATHEXTRA + '\\python-3.11.9-amd64.exe' + ' PrependPath=1')

    pr_green("Successfully Installed - Python 3.11.9")

    input("Press Enter to continue")
    os.system('cls')

    # install python packages from requirements.txt
    pr_cyan("Now let's install the python packages")
    appdataLocation = os.getenv("LOCALAPPDATA")

    os.environ['Path'] = ""
    os.environ['Path'] += f'{appdataLocation}\\Programs\\Python\\Python311\\Scripts'
    os.environ['Path'] += f';{appdataLocation}\\Programs\\Python\\Python311\\'

    os.system('python -m pip install --upgrade pip')
    os.system('pip install --upgrade setuptools')
    os.system(f'pip install -r {PATHEXTRA}\\requirements.txt')

    pr_green("Successfully Installed - Python Packages")

    input("Press Enter to continue")
    os.system('cls')

    # install CUDA Toolkit 12.6.0
    if selection != 1:
        pr_cyan("Now let's install CUDA Toolkit 12.6.0")

        os.system(PATHEXTRA + '\\cuda_12.6.0_windows_network.exe')

        pr_green(f"Successfully Installed - CUDA Toolkit 12.6.0")

        input("Press Enter to continue")
        os.system('cls')

    # pip install pytorch and onnx runtime (and cupy if Nvidia)
    # pytorch versions are carefully chosen out for compatibility
    pr_cyan("Now let's install PyTorch and ONNX Runtime (and cupy if Nvidia)")

    if selection != 1:
        os.system('pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121')
        os.system('pip install cupy-cuda12x')
        os.system('pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/')
    else:
        os.system('pip uninstall torch torchvision -y') # uninstalling potentially conflicting torch version that came with ultralytics package
        os.system('pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0')
        os.system('pip install onnxruntime-directml')

    pr_green(f"Successfully Installed - PyTorch and ONNX Runtime (and cupy if Nvidia)")
    pr_cyan("You are now ready to run the Fast and Faster version of the Aimbot")

    # nvidia fastest setup
    if selection != 1:
        pr_yellow("\nContinue to set up the Fastest Version (NVIDIA ONLY)...")
        
        input("Press Enter to continue")
        os.system('cls')

        # ask user to create an nvidia developer account
        pr_cyan("There next steps require you to make a FREE Nvidia Developer account")
        pr_green("Go Here to make an account: https://developer.nvidia.com/login")
        
        input("Press Enter to continue once you have made an account and logged in...")
        os.system('cls')

        # ask user to download zip files
        cuDNNFile = None
        TensorRTFile = None
        while cuDNNFile is None and TensorRTFile is None:
            if selection == 2:
                pr_cyan("Now let's download cuDNN 8.9.7 and TensorRT 8.6.1")
                pr_light_red("Make sure the zips are in your DOWNLOADS folder!")
                pr_yellow("Download cuDNN 8.9.7: https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.7/local_installers/12.x/cudnn-windows-x86_64-8.9.7.29_cuda12-archive.zip/")
                pr_yellow("Download TensorRT 8.6.1: https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/zip/TensorRT-8.6.1.6.Windows10.x86_64.cuda-12.0.zip/")
            elif selection == 3:
                pr_cyan("Now let's download cuDNN 9.3.0 and TensorRT 10.3.0")
                pr_light_red("Make sure the zips are in your DOWNLOADS folder!")
                pr_yellow("Download cuDNN 9.3.0: https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.3.0.75_cuda12-archive.zip/")
                pr_yellow("Download TensorRT 10.3.0: https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.3.0/zip/TensorRT-10.3.0.26.Windows.win10.cuda-12.5.zip/")
            
            pr_light_red("Do NOT RENAME or EXTRACT the zips")
            input("Press Enter to continue once you have downloaded the zips to your DOWNLOADS folder...")
            os.system('cls')

            # search for zip files in user downloads folder
            downloadsFolder = os.path.join(os.path.expanduser("~"), "Downloads")
            for file in os.listdir(downloadsFolder):
                if selection == 2:
                    if file == "cudnn-windows-x86_64-8.9.7.29_cuda12-archive.zip":
                        cuDNNFile = os.path.join(downloadsFolder, file)
                    if file == "TensorRT-8.6.1.6.Windows10.x86_64.cuda-12.0.zip":
                        TensorRTFile = os.path.join(downloadsFolder, file)
                elif selection == 3:
                    if file == "cudnn-windows-x86_64-9.3.0.75_cuda12-archive.zip":
                        cuDNNFile = os.path.join(downloadsFolder, file)
                    if file == "TensorRT-10.3.0.26.Windows.win10.cuda-12.5.zip":
                        TensorRTFile = os.path.join(downloadsFolder, file)
            
            # check if zip files found in user downloads directory
            if cuDNNFile is None or TensorRTFile is None:
                os.system('cls')
                pr_red("Could not find the required files in your Downloads folder, please ensure you have downloaded the zip files correctly")
        
        os.system('cls')
        
        # cuda 12.6.0 directory
        cudaDir = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6"

        # install cuDNN zip
        pr_cyan("Installing cuDNN zip file")
        extract_subfolder(cuDNNFile, cudaDir)
        pr_green("Installed cuDNN!")

        # install TensorRT zip
        pr_cyan("Installing TensorRT zip file")
        extract_subfolder(TensorRTFile, cudaDir)
        pr_green("Installed TensorRT!")

        # install tensorrt python package
        pr_cyan("Installing TensorRT python package")
        if selection == 2:
            os.system(f'pip install "{cudaDir}\\python\\tensorrt-8.6.1-cp311-none-win_amd64.whl"')
        elif selection == 3:
            os.system(f'pip install "{cudaDir}\\python\\tensorrt-10.3.0-cp311-none-win_amd64.whl"')
        pr_green("Installed TensorRT python package!")

        # adding CUDA lib to system PATH to prevent nvinfer.dll not found error
        newPath = f"{cudaDir}\\lib"
        systemPathKey = r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"

        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, systemPathKey, 0, winreg.KEY_READ | winreg.KEY_WRITE) as regKey:
                currentValue, regType = winreg.QueryValueEx(regKey, "Path")
                paths = currentValue.split(';')

                if newPath not in paths:
                    paths.append(newPath)
                    updatedValue = ';'.join(paths)
                    winreg.SetValueEx(regKey, "Path", 0, regType, updatedValue)
            pr_green("Added CUDA lib to system PATH")
        except Exception as e:
            print(f"Failed to add CUDA lib to system PATH: {e}")

        pr_cyan("You are now ready to run the Fastest version of the Aimbot, Have Fun!")
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        pr_red(f"Error: {e}")

    input("Press Enter to exit")
