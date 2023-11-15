import pygetwindow
import time
import bettercam

# Could be do with
# from config import *
# But we are writing it out for clarity for new devs
from config import aaRightShift, screenShotHeight, screenShotWidth

def gameSelection() -> (bettercam.BetterCam, int, int | None):
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
        return None

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
        return None
    print("Successfully activated the game window...")

    # Setting up the screen shots
    sctArea: dict[str, int] = {"mon": 1, "top": videoGameWindow.top + (videoGameWindow.height - screenShotHeight) // 2,
                         "left": aaRightShift + ((videoGameWindow.left + videoGameWindow.right) // 2) - (screenShotWidth // 2),
                         "width": screenShotWidth,
                         "height": screenShotHeight}

    #! Uncomment if you want to view the entire screen
    # sctArea = {"mon": 1, "top": 0, "left": 0, "width": 1920, "height": 1080}

    # Starting screenshoting engine
    left = aaRightShift + \
        ((videoGameWindow.left + videoGameWindow.right) // 2) - (screenShotWidth // 2)
    top = videoGameWindow.top + \
        (videoGameWindow.height - screenShotHeight) // 2
    right, bottom = left + screenShotWidth, top + screenShotHeight

    region: tuple = (left, top, right, bottom)

    # Calculating the center Autoaim box
    cWidth: int = sctArea["width"] / 2
    cHeight: int = sctArea["height"] / 2

    camera = bettercam.create(region=region, output_color="BGRA", max_buffer_len=512)
    if camera is None:
        print("Your Camera Failed! Ask @Wonder for help in our Discord in the #ai-aimbot channel ONLY: https://discord.gg/rootkitorg")
        return
    camera.start(target_fps=120, video_mode=True)

    return camera, cWidth, cHeight