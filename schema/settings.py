from pydantic import BaseModel, Field, validator
import win32con

KEY_MAP = {
    "F1": win32con.VK_F1,
    "F2": win32con.VK_F2,
    "F3": win32con.VK_F3,
    "F4": win32con.VK_F4,
    "F5": win32con.VK_F5,
    "F6": win32con.VK_F6,
    "F7": win32con.VK_F7,
    "F8": win32con.VK_F8,
    "F9": win32con.VK_F9,
    "F10": win32con.VK_F10,
    "F11": win32con.VK_F11,
    "F12": win32con.VK_F12,
    "Escape": win32con.VK_ESCAPE,
    "Tab": win32con.VK_TAB,
    "CapsLock": win32con.VK_CAPITAL,
    "LeftShift": win32con.VK_LSHIFT,
    "Shift": win32con.VK_LSHIFT,
    "RightShift": win32con.VK_RSHIFT,
    "LeftControl": win32con.VK_LCONTROL,
    "Control": win32con.VK_LCONTROL,
    "RightControl": win32con.VK_RCONTROL,
    "LeftAlt": win32con.VK_LMENU,
    "Alt": win32con.VK_LMENU,
    "RightAlt": win32con.VK_RMENU,
    "Enter": win32con.VK_RETURN,
    "Backspace": win32con.VK_BACK,
    "Delete": win32con.VK_DELETE,
    "Insert": win32con.VK_INSERT,
    "Home": win32con.VK_HOME,
    "End": win32con.VK_END,
    "PageUp": win32con.VK_PRIOR,
    "PageDown": win32con.VK_NEXT,
    "LeftMouseButtonDown": win32con.VK_LBUTTON,
    "RightMouseButtonDown": win32con.VK_RBUTTON,
    "MiddleMouseButtonDown": win32con.VK_MBUTTON,
}

class Settings(BaseModel):
    movementAmp: float = Field(default=0.3)
    useMask: bool = Field(default=False)
    maskLeft: bool = Field(default=True)
    maskWidth: int = Field(default=80)
    maskHeight: int = Field(default=200)
    quitKey: int = Field(default=ord("Q"))
    screenShotHeight: int = Field(default=320)
    screenShotWidth: int = Field(default=320)
    confidence: float = Field(default=0.5)
    headshotMode: bool = Field(default=True)
    headshotDistanceModifier: float = Field(default=0.38)
    displayCPS: bool = Field(default=True)
    visuals: bool = Field(default=False)
    centerOfScreen: bool = Field(default=True)
    activationKey: int = Field(default=win32con.VK_CAPITAL)
    autoFire: bool = Field(default=False)
    autoFireActivationDistance: int = Field(default=50)
    onnxChoice: int = Field(default=2)
    fovCircle: bool = Field(default=False)
    fovCircleRadius: int = Field(default=160)
    fovCircleRadiusDetectionModifier: float = Field(default=1.0)
    aimShakey: bool = Field(default=False)
    aimShakeyStrength: int = Field(default=10)
    toggleable: bool = Field(default=True)
    gameTitle: str = Field(default=None)

    @validator('activationKey', 'quitKey', pre=True)
    def mapKey(cls, key):

        if len(key) == 1:
            try:
                return ord(key.upper())
            except Exception as e:
                print("Invalid activation key")
                print("Defaulting to CapsLock")
                return KEY_MAP["CapsLock"]
            
        return KEY_MAP[key]

    class Config:
        extra = "allow"