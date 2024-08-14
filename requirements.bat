@echo off
setlocal

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed on this system.
    pause
    exit /b 1
)

echo Python is installed. Proceeding with package installations...

REM Create a log file
set log_file=install_log.txt
echo Installation started at %date% %time% > %log_file%

REM Install PyAutoGUI
echo Installing PyAutoGUI...
pip install PyAutoGUI >> %log_file% 2>&1
if %errorlevel% neq 0 (
    echo Failed to install PyAutoGUI
    pause
    exit /b %errorlevel%
)
echo Successfully installed PyAutoGUI

REM Install PyGetWindow
echo Installing PyGetWindow...
pip install PyGetWindow >> %log_file% 2>&1
if %errorlevel% neq 0 (
    echo Failed to install PyGetWindow
    pause
    exit /b %errorlevel%
)
echo Successfully installed PyGetWindow

REM Install opencv-python
echo Installing opencv-python...
pip install opencv-python >> %log_file% 2>&1
if %errorlevel% neq 0 (
    echo Failed to install opencv-python
    pause
    exit /b %errorlevel%
)
echo Successfully installed opencv-python

REM Install numpy
echo Installing numpy...
pip install numpy >> %log_file% 2>&1
if %errorlevel% neq 0 (
    echo Failed to install numpy
    pause
    exit /b %errorlevel%
)
echo Successfully installed numpy

REM Install pandas
echo Installing pandas...
pip install pandas >> %log_file% 2>&1
if %errorlevel% neq 0 (
    echo Failed to install pandas
    pause
    exit /b %errorlevel%
)
echo Successfully installed pandas

REM Install pywin32
echo Installing pywin32...
pip install pywin32 >> %log_file% 2>&1
if %errorlevel% neq 0 (
    echo Failed to install pywin32
    pause
    exit /b %errorlevel%
)
echo Successfully installed pywin32

REM Install pyyaml
echo Installing pyyaml...
pip install pyyaml >> %log_file% 2>&1
if %errorlevel% neq 0 (
    echo Failed to install pyyaml
    pause
    exit /b %errorlevel%
)
echo Successfully installed pyyaml

REM Install tqdm
echo Installing tqdm...
pip install tqdm >> %log_file% 2>&1
if %errorlevel% neq 0 (
    echo Failed to install tqdm
    pause
    exit /b %errorlevel%
)
echo Successfully installed tqdm

REM Install matplotlib
echo Installing matplotlib...
pip install matplotlib >> %log_file% 2>&1
if %errorlevel% neq 0 (
    echo Failed to install matplotlib
    pause
    exit /b %errorlevel%
)
echo Successfully installed matplotlib

REM Install seaborn
echo Installing seaborn...
pip install seaborn >> %log_file% 2>&1
if %errorlevel% neq 0 (
    echo Failed to install seaborn
    pause
    exit /b %errorlevel%
)
echo Successfully installed seaborn

REM Install requests
echo Installing requests...
pip install requests >> %log_file% 2>&1
if %errorlevel% neq 0 (
    echo Failed to install requests
    pause
    exit /b %errorlevel%
)
echo Successfully installed requests

REM Install ipython
echo Installing ipython...
pip install ipython >> %log_file% 2>&1
if %errorlevel% neq 0 (
    echo Failed to install ipython
    pause
    exit /b %errorlevel%
)
echo Successfully installed ipython

REM Install psutil
echo Installing psutil...
pip install psutil >> %log_file% 2>&1
if %errorlevel% neq 0 (
    echo Failed to install psutil
    pause
    exit /b %errorlevel%
)
echo Successfully installed psutil

REM Install dxcam
echo Installing dxcam...
pip install dxcam >> %log_file% 2>&1
if %errorlevel% neq 0 (
    echo Failed to install dxcam
    pause
    exit /b %errorlevel%
)
echo Successfully installed dxcam

REM Install onnxruntime-directml
echo Installing onnxruntime-directml...
pip install onnxruntime-directml >> %log_file% 2>&1
if %errorlevel% neq 0 (
    echo Failed to install onnxruntime-directml
    pause
    exit /b %errorlevel%
)
echo Successfully installed onnxruntime-directml

REM Install bettercam
echo Installing bettercam...
pip install bettercam >> %log_file% 2>&1
if %errorlevel% neq 0 (
    echo Failed to install bettercam
    pause
    exit /b %errorlevel%
)
echo Successfully installed bettercam

echo All packages installed successfully! See %log_file% for details.
pause
