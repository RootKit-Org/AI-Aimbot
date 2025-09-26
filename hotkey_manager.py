"""
热键管理系统
支持键盘和鼠标热键，组合键，以及运行时参数调整
"""
import win32api
import win32con
import time
from typing import Dict, Any, Callable, Optional

class GlobalHotkeyManager:
    """全局热键管理器"""
    
    def __init__(self, config_module):
        """
        初始化热键管理器
        
        Args:
            config_module: 配置模块对象
        """
        self.config = config_module
        self.hotkey_states = {}
        self.last_trigger_time = {}
        
        # 热键回调函数映射
        self.hotkey_callbacks = {}
        
        # 运行时可调参数的当前值
        self.runtime_values = {
            'sensitivity': getattr(config_module, 'aaMovementAmp', 0.75),
            'confidence': getattr(config_module, 'confidence', 0.4),
            'visuals_enabled': getattr(config_module, 'visuals', True),
            'debug_enabled': getattr(config_module, 'cpsDisplay', True),
            'aim_mode_advanced': getattr(config_module, 'enableAdvancedAiming', False),
            'headshot_mode': getattr(config_module, 'headshot_mode', False),
            'smooth_mode': getattr(config_module, 'enableSmoothMovement', True)
        }
        
        # 初始化热键配置
        self._init_hotkeys()
        
        print("[热键管理] 初始化完成")
        print("[热键管理] 支持的热键类型: 键盘按键, 鼠标按键, 组合键")
    
    def _init_hotkeys(self):
        """初始化热键配置和回调"""
        # 从配置中获取热键设置
        hotkeys_config = {
            'quit': getattr(self.config, 'aaQuitKey', 'P'),
            'toggle_visuals': getattr(self.config, 'aaToggleVisualsKey', 'V'),
            'toggle_debug': getattr(self.config, 'aaToggleDebugKey', 'F1'),
            'aim_mode_toggle': getattr(self.config, 'aimModeToggleKey', 'F2'),
            'headshot_toggle': getattr(self.config, 'headshotModeToggleKey', 'F3'),
            'smooth_toggle': getattr(self.config, 'smoothModeToggleKey', 'F4'),
            'sensitivity_up': getattr(self.config, 'sensitivityUpKey', 'NUMPAD_PLUS'),
            'sensitivity_down': getattr(self.config, 'sensitivityDownKey', 'NUMPAD_MINUS'),
            'confidence_up': getattr(self.config, 'confidenceUpKey', 'PAGEUP'),
            'confidence_down': getattr(self.config, 'confidenceDownKey', 'PAGEDOWN'),
            'emergency_stop': getattr(self.config, 'emergencyStopKey', 'CTRL+ESC'),
        }
        
        # 解析热键配置
        self.parsed_hotkeys = {}
        for name, key_str in hotkeys_config.items():
            if key_str and key_str.strip():
                self.parsed_hotkeys[name] = self._parse_hotkey(key_str)
        
        # 设置默认回调函数
        self._setup_default_callbacks()
    
    def _parse_hotkey(self, key_string: str) -> dict:
        """解析热键字符串"""
        if '+' in key_string:
            # 组合键处理
            keys = [k.strip().upper() for k in key_string.split('+')]
            result = {'modifiers': [], 'key': None, 'is_mouse': False}
            
            for key in keys:
                if key in ['CTRL', 'ALT', 'SHIFT']:
                    result['modifiers'].append(self._get_vk_code(key))
                else:
                    vk_code = self._get_vk_code(key)
                    result['key'] = vk_code & 0x7FFF  # 移除鼠标标记
                    result['is_mouse'] = bool(vk_code & 0x8000)
            
            return result
        else:
            # 单个按键
            vk_code = self._get_vk_code(key_string)
            return {
                'modifiers': [], 
                'key': vk_code & 0x7FFF,
                'is_mouse': bool(vk_code & 0x8000)
            }
    
    def _get_vk_code(self, key_name: str) -> int:
        """获取虚拟键码"""
        key_upper = key_name.upper()
        
        # 功能键
        function_keys = {
            "F1": win32con.VK_F1, "F2": win32con.VK_F2, "F3": win32con.VK_F3,
            "F4": win32con.VK_F4, "F5": win32con.VK_F5, "F6": win32con.VK_F6,
            "F7": win32con.VK_F7, "F8": win32con.VK_F8, "F9": win32con.VK_F9,
            "F10": win32con.VK_F10, "F11": win32con.VK_F11, "F12": win32con.VK_F12
        }
        
        # 特殊键
        special_keys = {
            "SPACE": win32con.VK_SPACE, "TAB": win32con.VK_TAB, "ENTER": win32con.VK_RETURN,
            "ESC": win32con.VK_ESCAPE, "CTRL": win32con.VK_CONTROL, "SHIFT": win32con.VK_SHIFT,
            "ALT": win32con.VK_MENU, "HOME": win32con.VK_HOME, "END": win32con.VK_END,
            "INSERT": win32con.VK_INSERT, "DELETE": win32con.VK_DELETE,
            "PAGEUP": win32con.VK_PRIOR, "PAGEDOWN": win32con.VK_NEXT,
            "UP": win32con.VK_UP, "DOWN": win32con.VK_DOWN,
            "LEFT": win32con.VK_LEFT, "RIGHT": win32con.VK_RIGHT,
            "CAPSLOCK": win32con.VK_CAPITAL
        }
        
        # 数字键盘
        numpad_keys = {
            "NUMPAD0": win32con.VK_NUMPAD0, "NUMPAD1": win32con.VK_NUMPAD1,
            "NUMPAD2": win32con.VK_NUMPAD2, "NUMPAD3": win32con.VK_NUMPAD3,
            "NUMPAD4": win32con.VK_NUMPAD4, "NUMPAD5": win32con.VK_NUMPAD5,
            "NUMPAD6": win32con.VK_NUMPAD6, "NUMPAD7": win32con.VK_NUMPAD7,
            "NUMPAD8": win32con.VK_NUMPAD8, "NUMPAD9": win32con.VK_NUMPAD9,
            "NUMPAD_PLUS": win32con.VK_ADD, "NUMPAD_MINUS": win32con.VK_SUBTRACT,
            "NUMPAD_MULTIPLY": win32con.VK_MULTIPLY, "NUMPAD_DIVIDE": win32con.VK_DIVIDE
        }
        
        # 鼠标按键（使用特殊标记）
        mouse_keys = {
            "MOUSE1": 0x01 | 0x8000,  # 左键 + 鼠标标记
            "MOUSE2": 0x02 | 0x8000,  # 右键 + 鼠标标记
            "MOUSE3": 0x04 | 0x8000,  # 中键 + 鼠标标记
            "MOUSE4": 0x05 | 0x8000,  # 侧键1 + 鼠标标记
            "MOUSE5": 0x06 | 0x8000   # 侧键2 + 鼠标标记
        }
        
        # 按优先级查找
        if key_upper in function_keys:
            return function_keys[key_upper]
        elif key_upper in special_keys:
            return special_keys[key_upper]
        elif key_upper in numpad_keys:
            return numpad_keys[key_upper]
        elif key_upper in mouse_keys:
            return mouse_keys[key_upper]
        elif len(key_name) == 1:
            return ord(key_upper)
        else:
            return ord('F')  # 默认值
    
    def _setup_default_callbacks(self):
        """设置默认热键回调函数"""
        
        def toggle_visuals():
            self.runtime_values['visuals_enabled'] = not self.runtime_values['visuals_enabled']
            print(f"[热键] 可视化显示: {'开启' if self.runtime_values['visuals_enabled'] else '关闭'}")
        
        def toggle_debug():
            self.runtime_values['debug_enabled'] = not self.runtime_values['debug_enabled']
            print(f"[热键] 调试信息: {'开启' if self.runtime_values['debug_enabled'] else '关闭'}")
        
        def toggle_aim_mode():
            self.runtime_values['aim_mode_advanced'] = not self.runtime_values['aim_mode_advanced']
            mode = '高级' if self.runtime_values['aim_mode_advanced'] else '普通'
            print(f"[热键] 瞄准模式: {mode}")
        
        def toggle_headshot_mode():
            self.runtime_values['headshot_mode'] = not self.runtime_values['headshot_mode']
            print(f"[热键] 爆头模式: {'开启' if self.runtime_values['headshot_mode'] else '关闭'}")
        
        def toggle_smooth_mode():
            self.runtime_values['smooth_mode'] = not self.runtime_values['smooth_mode']
            print(f"[热键] 平滑移动: {'开启' if self.runtime_values['smooth_mode'] else '关闭'}")
        
        def increase_sensitivity():
            step = getattr(self.config, 'sensitivityStep', 0.05)
            max_val = getattr(self.config, 'maxSensitivity', 2.0)
            self.runtime_values['sensitivity'] = min(max_val, self.runtime_values['sensitivity'] + step)
            print(f"[热键] 灵敏度: {self.runtime_values['sensitivity']:.3f}")
        
        def decrease_sensitivity():
            step = getattr(self.config, 'sensitivityStep', 0.05)
            min_val = getattr(self.config, 'minSensitivity', 0.1)
            self.runtime_values['sensitivity'] = max(min_val, self.runtime_values['sensitivity'] - step)
            print(f"[热键] 灵敏度: {self.runtime_values['sensitivity']:.3f}")
        
        def increase_confidence():
            step = getattr(self.config, 'confidenceStep', 0.05)
            max_val = getattr(self.config, 'maxConfidence', 0.9)
            self.runtime_values['confidence'] = min(max_val, self.runtime_values['confidence'] + step)
            print(f"[热键] 检测置信度: {self.runtime_values['confidence']:.3f}")
        
        def decrease_confidence():
            step = getattr(self.config, 'confidenceStep', 0.05)
            min_val = getattr(self.config, 'minConfidence', 0.1)
            self.runtime_values['confidence'] = max(min_val, self.runtime_values['confidence'] - step)
            print(f"[热键] 检测置信度: {self.runtime_values['confidence']:.3f}")
        
        def emergency_stop():
            print("[热键] 紧急停止！")
            # 这里可以添加紧急停止逻辑
        
        # 注册回调函数
        self.hotkey_callbacks = {
            'toggle_visuals': toggle_visuals,
            'toggle_debug': toggle_debug,
            'aim_mode_toggle': toggle_aim_mode,
            'headshot_toggle': toggle_headshot_mode,
            'smooth_toggle': toggle_smooth_mode,
            'sensitivity_up': increase_sensitivity,
            'sensitivity_down': decrease_sensitivity,
            'confidence_up': increase_confidence,
            'confidence_down': decrease_confidence,
            'emergency_stop': emergency_stop,
        }
    
    def check_hotkeys(self):
        """检查并处理所有热键"""
        if not getattr(self.config, 'enableHotkeys', True):
            return
        
        response_delay = getattr(self.config, 'hotkeyResponseDelay', 0.1)
        
        for name, key_config in self.parsed_hotkeys.items():
            if name in self.hotkey_callbacks and self._is_key_pressed(key_config, response_delay):
                try:
                    self.hotkey_callbacks[name]()
                except Exception as e:
                    print(f"[热键管理] 回调执行失败 '{name}': {e}")
    
    def _is_key_pressed(self, key_config: dict, response_delay: float = 0.1) -> bool:
        """检查热键是否被按下"""
        if not key_config or 'key' not in key_config:
            return False
        
        key_id = str(key_config)
        current_time = time.time()
        
        # 检查响应延迟
        if (key_id in self.last_trigger_time and 
            current_time - self.last_trigger_time[key_id] < response_delay):
            return False
        
        # 检查组合键
        for modifier in key_config.get('modifiers', []):
            if win32api.GetAsyncKeyState(modifier) == 0:
                self.hotkey_states[key_id] = False
                return False
        
        # 检查主键
        main_key = key_config['key']
        is_mouse = key_config.get('is_mouse', False)
        
        if is_mouse:
            # 鼠标按键检查
            if not getattr(self.config, 'enableMouseHotkeys', True):
                return False
            key_pressed = win32api.GetAsyncKeyState(main_key) != 0
        else:
            # 键盘按键检查
            key_pressed = win32api.GetAsyncKeyState(main_key) != 0
        
        # 检查按键状态变化（只在按下的瞬间触发）
        was_pressed = self.hotkey_states.get(key_id, False)
        self.hotkey_states[key_id] = key_pressed
        
        if key_pressed and not was_pressed:
            self.last_trigger_time[key_id] = current_time
            return True
        
        return False
    
    def is_activation_key_pressed(self) -> bool:
        """检查激活键是否被按下（用于瞄准激活）"""
        activation_key = getattr(self.config, 'aaActivationKey', 'CAPSLOCK')
        
        if activation_key == 'CAPSLOCK':
            # 大写锁定键特殊处理
            return bool(win32api.GetKeyState(win32con.VK_CAPITAL) & 0x8000)
        else:
            key_config = self._parse_hotkey(activation_key)
            return self._is_key_held(key_config)
    
    def is_mouse_activation_pressed(self) -> bool:
        """检查鼠标激活键是否被按下"""
        mouse_key = getattr(self.config, 'mouseActivationKey', '')
        if not mouse_key or not getattr(self.config, 'enableMouseHotkeys', True):
            return False
        
        key_config = self._parse_hotkey(mouse_key)
        return self._is_key_held(key_config)
    
    def _is_key_held(self, key_config: dict) -> bool:
        """检查热键是否持续按下"""
        if not key_config or 'key' not in key_config:
            return False
        
        # 检查组合键
        for modifier in key_config.get('modifiers', []):
            if win32api.GetAsyncKeyState(modifier) == 0:
                return False
        
        # 检查主键
        main_key = key_config['key']
        is_mouse = key_config.get('is_mouse', False)
        
        if is_mouse:
            return win32api.GetAsyncKeyState(main_key) != 0
        else:
            return win32api.GetAsyncKeyState(main_key) != 0
    
    def should_quit(self) -> bool:
        """检查是否应该退出程序"""
        if 'quit' in self.parsed_hotkeys:
            quit_config = self.parsed_hotkeys['quit']
            return self._is_key_held(quit_config)
        
        # 默认退出键检查
        quit_key = getattr(self.config, 'aaQuitKey', 'P')
        return win32api.GetAsyncKeyState(ord(quit_key.upper())) != 0
    
    def get_runtime_value(self, name: str, default=None):
        """获取运行时参数值"""
        return self.runtime_values.get(name, default)
    
    def update_config(self, new_config):
        """更新配置（用于热重载）"""
        self.config = new_config
        
        # 重新初始化热键
        self._init_hotkeys()
        
        # 更新运行时值
        self.runtime_values.update({
            'sensitivity': getattr(new_config, 'aaMovementAmp', self.runtime_values['sensitivity']),
            'confidence': getattr(new_config, 'confidence', self.runtime_values['confidence']),
        })
        
        print("[热键管理] 配置已更新")
    
    def print_hotkey_help(self):
        """打印热键帮助信息"""
        print("\n========== 热键说明 ==========")
        hotkey_help = {
            'aaActivationKey': '瞄准激活',
            'aaQuitKey': '退出程序',
            'aaToggleVisualsKey': '切换可视化',
            'aaToggleDebugKey': '切换调试信息',
            'aimModeToggleKey': '切换瞄准模式',
            'headshotModeToggleKey': '切换爆头模式',
            'smoothModeToggleKey': '切换平滑模式',
            'sensitivityUpKey': '增加灵敏度',
            'sensitivityDownKey': '减少灵敏度',
            'confidenceUpKey': '增加检测置信度',
            'confidenceDownKey': '减少检测置信度',
            'configReloadKey': '重载配置',
            'emergencyStopKey': '紧急停止',
        }
        
        for key_name, description in hotkey_help.items():
            key_value = getattr(self.config, key_name, '未设置')
            print(f"  {description}: {key_value}")
        
        print("==============================\n")

# 全局热键管理器实例
global_hotkey_manager: Optional[GlobalHotkeyManager] = None

def init_hotkey_manager(config_module):
    """初始化全局热键管理器"""
    global global_hotkey_manager
    global_hotkey_manager = GlobalHotkeyManager(config_module)
    return global_hotkey_manager

def get_hotkey_manager() -> Optional[GlobalHotkeyManager]:
    """获取全局热键管理器"""
    return global_hotkey_manager