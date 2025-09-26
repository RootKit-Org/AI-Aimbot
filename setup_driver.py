"""
AI-Aimbot 驱动级鼠标控制安装和配置脚本
"""
import os
import sys

def check_dll_file():
    """检查DLL文件是否存在"""
    dll_path = os.path.join(os.path.dirname(__file__), 'ghub_move', 'ghub_device.dll')
    
    if os.path.exists(dll_path):
        print("✅ ghub_device.dll 文件存在")
        return True
    else:
        print("❌ 找不到 ghub_device.dll 文件")
        print(f"应该位于: {dll_path}")
        return False

def check_python_modules():
    """检查必要的Python模块"""
    required_modules = ['ctypes', 'numpy', 'torch', 'cv2', 'pandas', 'win32api', 'win32con']
    missing_modules = []
    
    print("\n检查Python模块依赖...")
    
    for module in required_modules:
        try:
            if module == 'cv2':
                import cv2
            elif module == 'win32api':
                import win32api
            elif module == 'win32con':
                import win32con
            else:
                __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            missing_modules.append(module)
    
    return missing_modules

def check_mouse_driver():
    """测试鼠标驱动"""
    print("\n测试鼠标驱动...")
    try:
        from mouse_driver import is_driver_available, MouseDriver
        
        if is_driver_available():
            print("✅ 驱动级鼠标控制初始化成功!")
            return True
        else:
            print("❌ 驱动级鼠标控制初始化失败")
            print("可能的原因:")
            print("1. 未安装罗技驱动 (GHUB/LGS)")
            print("2. 安装的是新版本驱动，需要老版本驱动")
            print("3. DLL文件损坏或不兼容")
            return False
    except Exception as e:
        print(f"❌ 驱动测试失败: {e}")
        return False

def install_requirements():
    """安装缺失的依赖"""
    print("\n是否要自动安装缺失的Python模块? (y/n): ", end="")
    choice = input().lower().strip()
    
    if choice == 'y' or choice == 'yes':
        modules_to_install = {
            'cv2': 'opencv-python',
            'torch': 'torch',
            'numpy': 'numpy', 
            'pandas': 'pandas',
            'win32api': 'pywin32',
            'win32con': 'pywin32'
        }
        
        missing = check_python_modules()
        
        for module in missing:
            if module in modules_to_install:
                package = modules_to_install[module]
                print(f"\n安装 {package}...")
                os.system(f'pip install {package}')
    else:
        print("请手动安装缺失的模块")

def show_configuration_guide():
    """显示配置指南"""
    print("\n" + "="*60)
    print("配置指南")
    print("="*60)
    
    print("\n1. 罗技驱动设置:")
    print("   - 必须安装老版本罗技驱动 (GHUB/LGS)")
    print("   - 如果已安装新版本，请先完全卸载再安装老版本")
    
    print("\n2. 鼠标设置:")
    print("   - 打开Windows设置 -> 设备 -> 鼠标")
    print("   - 关闭 '提高鼠标精度' 选项")
    print("   - 将鼠标灵敏度调到正中间位置")
    
    print("\n3. 配置文件:")
    print("   - 编辑 config.py 文件")
    print("   - 设置 useDriverMouse = True")
    print("   - 设置 fallbackToSystemMouse = True (推荐)")
    
    print("\n4. 测试:")
    print("   - 运行 test_driver.py 进行功能测试")
    print("   - 或运行任何主程序文件测试实际效果")

def main():
    """主函数"""
    print("🎯 AI-Aimbot 驱动级鼠标控制安装配置助手")
    print("="*60)
    
    # 检查DLL文件
    dll_ok = check_dll_file()
    
    # 检查Python模块
    missing_modules = check_python_modules()
    
    if missing_modules:
        print(f"\n缺少以下模块: {', '.join(missing_modules)}")
        install_requirements()
    else:
        print("\n✅ 所有必要的Python模块都已安装")
    
    if dll_ok:
        # 测试驱动
        driver_ok = check_mouse_driver()
        
        if driver_ok:
            print("\n🎉 恭喜！驱动级鼠标控制配置成功！")
            print("你现在可以运行主程序享受驱动级鼠标控制了。")
        else:
            print("\n⚠️  驱动级鼠标控制配置未完成")
            print("请按照下面的配置指南完成设置")
    
    # 显示配置指南
    show_configuration_guide()
    
    print("\n" + "="*60)
    print("安装配置完成！")
    
    if dll_ok and not missing_modules:
        print("\n可以运行以下命令进行测试:")
        print("  python test_driver.py     # 驱动功能测试")
        print("  python main.py           # 运行主程序")
    
    input("\n按回车键退出...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n\n程序发生错误: {e}")
        import traceback
        traceback.print_exc()
        input("\n按回车键退出...")