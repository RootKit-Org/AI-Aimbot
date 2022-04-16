[![Pull Requests Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](http://makeapullrequest.com)
# AI Aimbot

<p float="left">
    <img src="imgs/halo.PNG" width="200" />
    <img src="imgs/valorant.PNG" width="200" /> 
    <img src="imgs/csgo.PNG" width="200" />
</p>

## Table of Contents
  - [Intro](#intro)
  - [Configurable Settings](#configurable-settings)
  - [Current Stats](#current-stats)
    - [REQUIREMENTS](#requirements)
    - [Pre-setup](#pre-setup)
    - [Run](#run)
  - [Community Based](#community-based)

## Intro
AI Aimbot works in any game with humanoid characters and utilizes YOLOv5. (Mouse movements don't work in Valorant currently) It is currently 100% undetectable to anti-cheat systems due to it being visual-only in nature. You could be found cheating if you are manually reported by another player and your game is reviewed manually due to the botty looking aimming of an Aimbot.

This is meant for educational purposed and to expose how poorly prepared game developers are with these new waves on AI based cheats. Please share with this with your friendly neighborhood game dev so they can start patching.

Code is all in 1 file for easy of viewing and learning from.

***Use at your own risk. If you get banned get rekted idiot***

**Adhere to our GNU licence, come on we are a nonprofit.**<br />
- free to use, sell, profit from, litterally anything you want to do with it
- **credit MUST be given to RootKit for the underlying base code**

Watch the tutorial video! - Coming Soon<br />
Watch the shorts video! - https://youtu.be/EEgspHlU_H0

Join teh Discord - https://discord.gg/rootkit

## Configurable Settings
*Default settings are good for most use cases. Read comments in code for more details.<br>
**CAPS_LOCK is the default for toggling on and off the autoaim functionality**

`videoGameWindowTitle` - (CHANGES PER GAME) Window title of the game you want to play. Does not need to be the complete window title.

`aaRightShift` - May need to be changed in 3rd person games like Fortnite and New World. Typically `100` or `150` will be sufficient.

`aaQuitKey` - Default is `q`, this may need to be changed to another key depending on the game.

`headshot_mode` - Make `False` if you want to aim more toward center mass.

`cpsDisplay` - Make `False` if you don't want the CPS to be displayed in the terminal.

`visuals` - Make `True` if you want to see what the AI sees. Can help with debugging issues.

`aaMovementAmp` - Default should be fine for 99% of use cases. Lower the value, the more smooth the autoaim will be. Recommended range is `0.5` - `2`.

`confidence` - Default should be kept unless you know what you are doing.

`screenShotHeight` - Default should be kept unless you know what you are doing.

`screenShotWidth` - Default should be kept unless you know what you are doing.

`aaDetectionBox` - Default should be kept unless you know what you are doing.

## Current Stats
This bot's speed is VERY dependent on your hardware. We will update the model it uses for detection later with a faster one.

Bot was tested on a:
- AMD Ryzen 7 2700
- 64 GB DDR4
- Nvidia RTX 2080

We got anywhere from 20-45 corrections per second. All games were ran at 1920x1080 or close to it when testing.

ANYTHING dealing with Machine Learning can be funky with your computer. So if you keep getting CUDA errors, you may want to restart your PC in order to make sure everything resets properly.

### REQUIREMENTS
- Nvidia RTX 2050 or higher
- Nvidia CUDA Toolkit 11.3 (https://developer.nvidia.com/cuda-11.3.0-download-archive)

### Pre-setup
1. Unzip the file and place the folder somewhere easy to access

2. Make sure you have a pet Python (aka install python) - https://www.python.org/

3. (Windows Users) Open up either `PowerShell` or `Command Prompt`. This can be done by pressing the Windows Key and searching for one of those applications.

4. To install `PyTorch` go to this website, https://pytorch.org/get-started/locally/, and Select the stable build, your OS, Pip, Python and CUDA 11.3. Then select the text that is generated and run that command.

5. Copy and past the command below into your terminal. This will install the Open Source packages needed to run the program. You will need to `cd` into the downloaded directory first. Follow step 2 in the Run section below if you need help.

```
pip install -r requirements.txt
```

### Run
If you have python and the packages you are good to go. Load up any game on your MAIN monitor and load into a game.

1. (Windows Users) Open up either `PowerShell` or `Command Prompt`. This can be done by pressing the Windows Key and searching for one of those applications.

2. Type `cd ` (make sure you add the space after the cd or else I will call you a monkey)

3. Drag and drop the folder that has the bot code onto the terminal

4. Press the enter key

5. Type `python main.py`, press enter.

6. Use CAPS_LOCK to toggle on and off the autoaim functionality. **It is off by defualt**

7. Pressing `q` at anytime will completely quit the program

## Community Based
We are a community based nonprofit. We are always open to pull requests on any of our repos. You will always be given credit for all of you work. Depending on what you contribute, we will give you any revenue earned on your contributions 💰💰💰!

**We are always looking for new Volunteers to join our Champions!
If you have any ideas for videos or programs, let us know!**
