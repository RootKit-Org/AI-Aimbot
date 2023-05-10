import dxcam
import math
import multiprocessing as mp
import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
from pynput import keyboard
import pygame
import time
from ultralytics import YOLO
import win32api, win32con, win32gui

global finished
finished = False
def on_press(key):
    global finished
    if key == keyboard.Key.f2:
        print("\n[INFO] F2 WAS PRESSED. QUITTING...")
        finished = True

def p_engine(q_target, q_view, model_name, dims):
    box_left, box_top, box_size, crosshair_x, crosshair_y = dims
    region = (box_left, box_top, box_left + box_size, box_top + box_size)
    camera = dxcam.create(region=region, output_color="RGB")
    camera.start(target_fps=120, video_mode=True) #video_mode=false will wait for unique frames before returning
    model = YOLO(model_name)
    while 1:
        frame = camera.get_latest_frame()
        results = model(frame, stream=True, device=0, classes=0, conf=.35, iou=.25, verbose=False)
        best_detection = float("inf")
        detections, target = [], (0,0,0)
        for result in results:
            for box in result.boxes: #iterate over each detection
                xyxy, confidence = box.xyxy[0].tolist(), float(box.conf[0].item())
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                head_x, head_y, width, height = x1 + int((x2 - x1) / 2), y1 + int(.2 * (x2 - x1)), x2 - x1, y2 - y1
                rel_x, rel_y = head_x - crosshair_x, head_y - crosshair_y
                this_detection = math.dist((0, 0), (rel_x, rel_y))
                detections.append((x1, y1, head_x, head_y, confidence, width, height))
                    
                if this_detection < best_detection:
                    best_detection = this_detection
                    target = True, rel_x, rel_y

        q_target.put(target)
        q_view.put((detections, target))
            
def p_view(q_view, dims):
    box_left, box_top, box_size, crosshair_x, crosshair_y = dims
    pygame.font.init()
    font = pygame.font.SysFont("freesansbold", 36)
    chromakey, dark_red = (0, 0, 0), (255, 100, 0)
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    hwnd = pygame.display.get_wm_info()["window"]
    win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE) | win32con.WS_EX_LAYERED | win32con.WS_EX_NOACTIVATE)
    win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(*chromakey), 0, win32con.LWA_COLORKEY)
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, box_left - 40, box_top - 40, 720, 720, 0)
    screen.fill(chromakey)
    pygame.display.update()
    
    while 1:
        pygame.event.pump()
        try:
            detections, target = q_view.get(timeout=1)
        except:
            detections, target = [], (0,0,0)
        
        in_view, x, y = target
        pygame.draw.line(screen, dark_red, (crosshair_x + 40 + x, crosshair_y + 40 + y), (crosshair_x + 40, crosshair_y + 40), 2)
        for target in detections:
            x1, y1, head_x, head_y, confidence, width, height = target
            #pygame.draw.circle(screen, dark_red, (40 + head_x, 40 + head_y), .4 * width, 1)
            #pygame.draw.circle(screen, dark_red, (40 + head_x, 40 + head_y), 6)
            pygame.draw.rect(screen, dark_red, pygame.Rect(36 + x1, 36 + y1, width + 8, height + 8), 2)
            text = font.render(f"{int(confidence * 20)*5}", 0, dark_red)
            screen.blit(text, (30 + x1, 10 + y1))

        pygame.draw.rect(screen, dark_red, pygame.Rect(36, 36, box_size+8, box_size+8), 1)
        pygame.display.update()
        screen.fill(chromakey)

if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n[INFO] PRESS F2 TO QUIT.\n")
    
    keyboard.Listener(on_press=on_press).start()    
    hsw = int(win32api.GetSystemMetrics(0)/2)
    hsh = int(win32api.GetSystemMetrics(1)/2)
    box_size = 640
    box_left = hsw - box_size//2
    box_top = hsh - box_size//2
    crosshair_x, crosshair_y = hsw - box_left, hsh - box_top
    dims = box_left, box_top, box_size, crosshair_x, crosshair_y
    q_target, q_view = mp.Queue(), mp.Queue()
    engine = mp.Process(target=p_engine, args=(q_target, q_view, 'yolov8s.pt', dims))
    view = mp.Process(target=p_view, args=(q_view, dims))
    engine.start()
    view.start()
        
    print_tick = fpsCount = tpsCount = 0
    while not finished:        
        try:
            target = q_target.get(timeout=1)
            fpsCount+=1
        except:
            target = 0, 0, 0

        tick = time.perf_counter()
        in_view, x, y = target

        if in_view:
            tpsCount+=1

        if tick - print_tick > 1 and fpsCount > 0:
            print_tick = tick
            print('TPS/FPS:', tpsCount, '/', fpsCount)
            fpsCount = tpsCount = 0

    #F2 pressed, loop finished, cleanup
    engine.terminate()
    view.terminate()
    os._exit(0)
    
