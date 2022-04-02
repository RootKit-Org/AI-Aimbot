[![Pull Requests Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](http://makeapullrequest.com)
# AI Aimbot
**Adhere to our GNU licence!!!**<br />
- free to use, sell, profit from, litterally anything you want to do with it
- **credit MUST be given to RootKit for the underlying base code**

Watch the tutorial video! - Coming Soon<br />
Watch the shorts video! - https://youtu.be/EEgspHlU_H0

Join teh Discord - https://discord.gg/rootkit

## Current Stats
This bot's speed is VERY dependent on your hardware. We will update the model it uses for detection later with a faster one.

Bot was tested on a:
- AMD Ryzen 7 2700
- 64 GB DDR4
- Nvidia RTX 2080

We got anywhere from 15-35 corrections per second. All games were ran at 1280x720 or close to it when testing.

ANYTHING dealing with Machine Learning can be funky with your computer. So if you keep getting CUDA errors, you may want to restart your PC in order to make sure everything resets properly.

### REQUIREMENTS
- Nvidia RTX 2050 or higher
- Nvidia CUDA Toolkit 11.3 (https://developer.nvidia.com/cuda-11.3.0-download-archive)

### Pre-setup
1. Unzip the file and place the folder somewhere easy to access

2. Make sure you have a pet Python (aka install python) - https://www.python.org/

3. (Windows Users) Open up either `PowerShell` or `Command Prompt`. This can be done by pressing the Windows Key and searching for one of those applications.

4. To install `PyTorch` go to this website, https://pytorch.org/get-started/locally/, and Select the stable build, your OS, Pip, Python and CUDA 11.3. Then select the text that is generated and run that command.

6. Copy and past the commands below into your terminal. This will install the Open Source packages needed to run the program.
```
pip install PyAutoGUI
pip install PyDirectInput
pip install Pillow
pip install opencv-python
pip install mss
pip install numpy
pip install pandas
pip install pywin32
pip install pyyaml
pip install tqdm
pip install matplotlib
pip install seaborn
```

### Run
If you have python and the packages you are good to go. Load up any game on your MAIN monitor and load into a game.

1. (Windows Users) Open up either `PowerShell` or `Command Prompt`. This can be done by pressing the Windows Key and searching for one of those applications.

2. Type `cd ` (make sure you add the space after the cd or else I will call you a monkey)

3. Drag and drop the folder that has the bot code onto the terminal

4. Press the enter key

5. Type `python main.py`, press enter and that is it! **IF YOU PRESS Q IT WILL STOP THE PROGRAM**

## Community Based
We are a community based nonprofit. We are always open to pull requests on any of our repos. You will always be given credit for all of you work. Depending on what you contribute, we will give you any revenue earned on your contributions 💰💰💰!

**We are always looking for new Volunteers to join our Champions!
If you have any ideas for videos or programs, let us know!**
