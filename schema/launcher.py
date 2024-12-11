from pydantic import BaseModel, Field, validator
from typing import Any

class Settings(BaseModel):
    name: str = Field(default="My Game Settings")
    groups: list["Group"] = Field(default=[])
    version: int = Field(default=1)

class UniversalBase(BaseModel):
    name: str = Field(default="")
    description: str = Field(default="")
    default: Any = Field(default=False)
    tierLock: int = Field(default=0)
    element: str = Field(default="")

    class Config:
        arbitrary_types_allowed=True

class Group(UniversalBase):
    items: list[Any] = Field(default=[])

class Slider(UniversalBase):
    type: str = Field(default="int")
    min: float = Field(default=0)
    max: float = Field(default=100)
    step: float = Field(default=1)
    value: float | None = Field(default=None)
    element: str = Field(default="slider")

class Toggle(UniversalBase):
    value: bool | None = Field(default=None)
    element: str = Field(default="toggle")

class Dropdown(UniversalBase):
    filePath: str = Field(default="")
    filter: str = Field(default="")
    options: list[str] = Field(default=[])
    value: str | None = Field(default=None)
    element: str = Field(default="dropdown")

class Keybind(UniversalBase):
    type: str = Field(default="str")
    value: str | None = Field(default=None)
    element: str = Field(default="keybind")

class TextField(UniversalBase):
    value: str | None = Field(default=None)
    element: str = Field(default="text")

class NumberField(UniversalBase):
    min: int | float = Field(default=0)
    max: int | float = Field(default=100)
    step: int | float = Field(default=1)
    value: int | float | None = Field(default=None)
    element: str = Field(default="number")



settings = Settings(
    name="RootKit Aimbot Settings"
)

settings.groups.append(
    Group(
        name="Basic Settings",
        description="Explore and customize!",
        items=[
            Slider(
                name="Mouse Movement Amplifier",
                description="",
                default=0.4,
                min=0.1,
                max=1.0,
                step=0.1
            ),
            Slider(
                name="Confidence",
                description="",
                default=0.4,
                min=0.1,
                max=1.0,
                step=0.1
            ),
            Toggle(
                name="Display CPS",
                default=True
            ),
            Toggle(
                name="Display Visuals",
                description="TURN OFF WHEN USING. Good for debuging, Display visuals such as the aimbot's target and the center of the screen",
                default=False
            ),
            Toggle(
                name="Prio Center People",
                description="Prioritize people who are closest to the center of the screen",
                default=True
            ),
            Keybind(
                name="Quit Key",
                description="The key that quits the aimbot",
                default="Q"
            ),
            Keybind(
                name="Activation Key",
                description="The key that activates the aimbot",
                default="CapsLock"
            ),
            Dropdown(
                name="Hardware Selection (Faster ONLY)",
                description="Choose the ONNX model to use",
                options=["CPU", "AMD", "NVidia"],
                default="AMD"
            ),
            Toggle(
                name="Headshot Mode",
                description="Prioritize headshots",
                default=True
            ),
            Slider(
                name="Headshot Distance Modifier",
                description="The distance modifier for headshots",
                default=0.38,
                min=0.1,
                max=1.0,
                step=0.1
            ),
            Toggle(
                name="Auto Fire",
                description="Automatically fire when a target is detected",
                default=False
            ),
            Slider(
                name="Auto Fire Activation Distance",
                description="The distance at which the aimbot will automatically fire",
                default=50,
                min=10,
                max=100,
                step=10
            ),
            Toggle(
                name="Toggleable Aimbot",
                description="Change between toggleable and holdable aimbot",
                default=True
            ),
            TextField(
                name="Game Title (Optional)",
                description="The title of the game",
                default="Change Me"
            )
        ]
    )
)

settings.groups.append(
    Group(
        name="Mask Settings",
        description="Use in 3rd person games to hide your Player Model",
        items=[
            Toggle(
                name="Active",
                description="Use a mask to hide your player model",
                default=False
            ),
            Toggle(
                name="Left",
                description="Should the mask be on the left or right side of the screen?",
                default=True
            ),
            Slider(
                name="Mask Width",
                description="The width of the mask",
                default=80,
                min=1,
                max=640,
                step=10
            ),
            Slider(
                name="Mask Height",
                description="The height of the mask",
                default=200,
                min=1,
                max=640,
                step=10
            )
        ]
    )
)

settings.groups.append(
    Group(
        name="FOV Settings",
        description="Change the circular FOV of the bot",
        items=[
            Toggle(
                name="Active",
                description="Draw a circle around the center of the screen",
                default=False
            ),
            Slider(
                name="Radius (Pixels)",
                description="The radius of the circle",
                default=160,
                min=1,
                max=320,
                step=1
            ),
            Slider(
                name="Detection Modifier",
                description="The detection modifier for the circle",
                default=1.0,
                min=0.1,
                max=2.0,
                step=0.1
            )
        ]
    )
)

settings.groups.append(
    Group(
        name="Aim Shake Settings",
        description="Add artificatial aim shake the aimbot",
        items=[
            Toggle(
                name="Active",
                description="Shake the aimbot's target",
                default=False
            ),
            Slider(
                name="Strength",
                description="The strength of the shake",
                default=10,
                min=1,
                max=100,
                step=1
            )
        ]
    )
)

settings.groups.append(
    Group(
        name="AI Vision",
        description="320 is recommened. 640 is max. Only increase if you have a high end GPU",
        items=[
            Slider(
                name="Scheenshot Width",
                description="How wide the screenshot is",
                default=320,
                min=1,
                max=640,
                step=1
            ),
            Slider(
                name="Scheenshot Height",
                description="How tall the screenshot is",
                default=320,
                min=1,
                max=640,
                step=1
            )
        ]
    )
)

print(settings.model_dump_json())
