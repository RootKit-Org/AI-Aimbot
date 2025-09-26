"""
驱动级鼠标控制模块
使用ghub_device.dll实现驱动级鼠标移动
"""
import ctypes
import os
from ctypes import CDLL

class MouseDriver:
    def __init__(self):
        """初始化驱动级鼠标控制"""
        self.driver = None
        self.is_initialized = False
        self._initialize_driver()
    
    def _initialize_driver(self):
        """初始化罗技驱动"""
        try:
            # 获取当前目录下的dll文件路径
            dll_path = os.path.join(os.path.dirname(__file__), 'ghub_move', 'ghub_device.dll')
            
            if not os.path.exists(dll_path):
                print(f'警告: 找不到驱动文件 {dll_path}')
                return
            
            self.driver = CDLL(dll_path)
            
            # 尝试初始化驱动
            init_result = self.driver.device_open()
            if init_result == 1:
                self.is_initialized = True
                print('驱动级鼠标控制初始化成功!')
            else:
                print('驱动初始化失败: 未安装GHUB或LGS驱动!')
                
        except FileNotFoundError:
            print('错误: 缺少ghub_device.dll文件')
        except Exception as e:
            print(f'驱动初始化异常: {e}')
    
    def move_relative(self, x, y):
        """
        相对移动鼠标
        
        Args:
            x (int): 水平移动距离，正数向右，负数向左
            y (int): 垂直移动距离，正数向下，负数向上
        """
        if not self.is_initialized:
            return False
        
        try:
            # 使用驱动级移动
            self.driver.moveR(int(x), int(y), False)
            return True
        except Exception as e:
            print(f'鼠标移动失败: {e}')
            return False
    
    def move_absolute(self, x, y):
        """
        绝对移动鼠标
        
        Args:
            x (int): 绝对X坐标
            y (int): 绝对Y坐标
        """
        if not self.is_initialized:
            return False
        
        try:
            # 使用驱动级绝对移动
            self.driver.moveR(int(x), int(y), True)
            return True
        except Exception as e:
            print(f'鼠标绝对移动失败: {e}')
            return False
    
    def click_left_down(self):
        """左键按下"""
        if not self.is_initialized:
            return False
        
        try:
            self.driver.mouse_down(1)  # 1代表左键
            return True
        except Exception as e:
            print(f'左键按下失败: {e}')
            return False
    
    def click_left_up(self):
        """左键松开"""
        if not self.is_initialized:
            return False
        
        try:
            self.driver.mouse_up(1)  # 1代表左键
            return True
        except Exception as e:
            print(f'左键松开失败: {e}')
            return False
    
    def click_left(self):
        """左键点击"""
        if not self.is_initialized:
            return False
        
        self.click_left_down()
        self.click_left_up()
        return True
    
    def click_right_down(self):
        """右键按下"""
        if not self.is_initialized:
            return False
        
        try:
            self.driver.mouse_down(3)  # 3代表右键
            return True
        except Exception as e:
            print(f'右键按下失败: {e}')
            return False
    
    def click_right_up(self):
        """右键松开"""
        if not self.is_initialized:
            return False
        
        try:
            self.driver.mouse_up(3)  # 3代表右键
            return True
        except Exception as e:
            print(f'右键松开失败: {e}')
            return False
    
    def click_right(self):
        """右键点击"""
        if not self.is_initialized:
            return False
        
        self.click_right_down()
        self.click_right_up()
        return True
    
    def is_driver_ready(self):
        """检查驱动是否准备就绪"""
        return self.is_initialized
    
    def close(self):
        """关闭驱动连接"""
        if self.driver and self.is_initialized:
            try:
                # 如果有关闭函数的话调用
                pass
            except:
                pass
        self.is_initialized = False

# 创建全局鼠标驱动实例
mouse_driver = MouseDriver()

def move_mouse_relative(x, y):
    """
    相对移动鼠标的便捷函数
    
    Args:
        x (int): 水平移动距离
        y (int): 垂直移动距离
    
    Returns:
        bool: 是否移动成功
    """
    return mouse_driver.move_relative(x, y)

def move_mouse_absolute(x, y):
    """
    绝对移动鼠标的便捷函数
    
    Args:
        x (int): 绝对X坐标
        y (int): 绝对Y坐标
    
    Returns:
        bool: 是否移动成功
    """
    return mouse_driver.move_absolute(x, y)

def click_mouse_left():
    """左键点击便捷函数"""
    return mouse_driver.click_left()

def click_mouse_right():
    """右键点击便捷函数"""
    return mouse_driver.click_right()

def is_driver_available():
    """检查驱动是否可用"""
    return mouse_driver.is_driver_ready()