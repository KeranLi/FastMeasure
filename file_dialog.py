import sys
import os

def select_image(title="选择图片"):
    """
    跨平台图片选择对话框
    支持: Windows, macOS, Linux
    """
    # 支持的图片格式
    filetypes = [
        ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.gif"),
        ("All files", "*.*")
    ]
    
    if sys.platform == 'darwin':  # macOS
        return _select_image_mac(title, filetypes)
    else:  # Windows & Linux
        return _select_image_tk(title, filetypes)

def _select_image_mac(title, filetypes):
    """macOS专用：避免子线程问题"""
    try:
        # 方案1: 使用PyQt5/6（推荐，更稳定）
        try:
            from PyQt6.QtWidgets import QApplication, QFileDialog
            app = QApplication.instance() or QApplication(sys.argv)
            dialog = QFileDialog()
            dialog.setWindowTitle(title)
            dialog.setNameFilters([f"{desc} ({pattern})" for desc, pattern in filetypes])
            dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
            
            if dialog.exec() == QFileDialog.DialogCode.Accepted:
                return dialog.selectedFiles()[0]
            return None
            
        except ImportError:
            pass
        
        # 方案2: 使用AppleScript（原生体验）
        try:
            import subprocess
            script = '''
            tell application "System Events"
                activate
                set imageFile to choose file with prompt "{}" of type {{"jpg", "jpeg", "png", "tif", "tiff", "bmp", "gif"}}
                return POSIX path of imageFile
            end tell
            '''.format(title)
            
            result = subprocess.run(['osascript', '-e', script], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
            return None
            
        except Exception:
            pass
        
        # 方案3: 回退到命令行输入
        return _fallback_cli_input(title)
        
    except Exception as e:
        print(f"macOS文件选择失败: {e}")
        return _fallback_cli_input(title)

def _select_image_tk(title, filetypes):
    """Windows/Linux: 标准Tkinter"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        # 确保在主线程
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)  # 置顶
        
        path = filedialog.askopenfilename(
            title=title,
            filetypes=filetypes
        )
        root.destroy()
        return path if path else None
        
    except Exception as e:
        print(f"Tkinter文件选择失败: {e}")
        return _fallback_cli_input(title)

def _fallback_cli_input(title):
    """终极回退：命令行输入"""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print("提示: 可将图片拖拽到终端窗口自动填充路径")
    print("      或手动输入完整路径")
    print(f"{'='*50}")
    
    try:
        path = input("图片路径: ").strip().strip("'\"")
        if os.path.exists(path):
            return path
        else:
            print(f"路径不存在: {path}")
            return None
    except KeyboardInterrupt:
        return None

# 测试
if __name__ == "__main__":
    result = select_image("测试选择图片")
    print(f"选择结果: {result}")