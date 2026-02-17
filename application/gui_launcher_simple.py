#!/usr/bin/env python
"""
FastMeasure Simple GUI Launcher
完全独立的打包应用，不依赖系统 Python
"""

import sys
import os
import subprocess
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from datetime import datetime

# 检测是否在 PyInstaller 打包环境中运行
def get_resource_path():
    """获取资源路径（打包后或开发环境）"""
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    else:
        return Path(__file__).parent

def is_frozen_app():
    """检查是否在打包后的应用中运行"""
    return getattr(sys, 'frozen', False)

# 全局资源路径
RESOURCE_PATH = get_resource_path()


def get_executable_path():
    """获取打包后的可执行文件路径"""
    if is_frozen_app():
        # 打包后，返回应用主可执行文件
        if sys.platform == "darwin":
            # macOS: FastMeasure.app/Contents/MacOS/FastMeasure
            return Path(sys.executable)
        else:
            return Path(sys.executable)
    else:
        # 开发环境，使用 Python
        return Path(sys.executable)


class FastMeasureSimpleGUI:
    """Simplified GUI that launches external processes."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("FastMeasure - Rock Grain Segmentation")
        self.root.geometry("700x600")
        self.root.minsize(600, 500)
        
        # Variables
        self.input_path = tk.StringVar()
        self.model_var = tk.StringVar(value="fastsam")
        self.mode_var = tk.StringVar(value="auto")
        self.status_var = tk.StringVar(value="Ready")
        
        self.process = None
        self.is_running = False
        
        self._create_ui()
        self._check_models()
    
    def _create_ui(self):
        """Create user interface."""
        # Title
        ttk.Label(self.root, text="FastMeasure", font=("Helvetica", 20, "bold")).pack(pady=10)
        ttk.Label(self.root, text="Rock Grain Auto Segmentation System").pack()
        
        # Main frame
        frame = ttk.Frame(self.root, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Input selection
        input_frame = ttk.LabelFrame(frame, text="Input", padding="10")
        input_frame.pack(fill=tk.X, pady=5)
        
        ttk.Entry(input_frame, textvariable=self.input_path).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(input_frame, text="Image", command=self._select_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(input_frame, text="Folder", command=self._select_folder).pack(side=tk.LEFT, padx=2)
        
        # Settings
        settings_frame = ttk.LabelFrame(frame, text="Settings", padding="10")
        settings_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(settings_frame, text="Model:").grid(row=0, column=0, sticky=tk.W)
        ttk.Combobox(settings_frame, textvariable=self.model_var, values=["fastsam", "mobilesam"], state="readonly", width=12).grid(row=0, column=1, padx=5)
        
        ttk.Label(settings_frame, text="Mode:").grid(row=0, column=2, sticky=tk.W, padx=(20, 5))
        ttk.Radiobutton(settings_frame, text="Auto", variable=self.mode_var, value="auto").grid(row=0, column=3)
        ttk.Radiobutton(settings_frame, text="Batch", variable=self.mode_var, value="batch").grid(row=0, column=4)
        ttk.Radiobutton(settings_frame, text="Interactive", variable=self.mode_var, value="interactive").grid(row=0, column=5)
        
        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(pady=10)
        
        self.run_btn = ttk.Button(btn_frame, text="▶ Run", command=self._run, width=15)
        self.run_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="⏹ Stop", command=self._stop, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="📁 Results", command=self._open_results, width=12).pack(side=tk.LEFT, padx=5)
        
        # Log
        log_frame = ttk.LabelFrame(frame, text="Log", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=15, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        ttk.Label(frame, textvariable=self.status_var).pack(anchor=tk.W)
        
        self._log("FastMeasure Ready")
        self._log("Select an image and click 'Run' to start")
        
        if is_frozen_app():
            self._log("ℹ️ Running as packaged app (no Python required)")
    
    def _check_models(self):
        """Check if models exist."""
        models_dirs = [
            Path("models"),
            RESOURCE_PATH / "models",
            RESOURCE_PATH.parent / "models",
        ]
        
        found = False
        for models_dir in models_dirs:
            if models_dir.exists() and list(models_dir.glob("*.pt")):
                found = True
                self._log(f"✓ Found models in: {models_dir}")
                break
        
        if not found:
            self._log("⚠ Warning: No models found in ./models/")
            self._log("Please place model files in the 'models' folder")
    
    def _log(self, message):
        """Add log message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
    
    def _select_image(self):
        """Select image file."""
        path = filedialog.askopenfilename(filetypes=[("Images", "*.tif *.tiff *.jpg *.png")])
        if path:
            self.input_path.set(path)
            self.mode_var.set("auto")
            self._log(f"Selected: {path}")
    
    def _select_folder(self):
        """Select folder."""
        path = filedialog.askdirectory()
        if path:
            self.input_path.set(path)
            self.mode_var.set("batch")
            self._log(f"Selected folder: {path}")
    
    def _build_command(self):
        """构建运行命令"""
        model = self.model_var.get()
        mode = self.mode_var.get()
        input_path = self.input_path.get()
        
        # Interactive 模式需要图片
        if mode == "interactive":
            if not input_path:
                filetypes = [("Images", "*.tif *.tiff *.jpg *.jpeg *.png *.bmp")]
                path = filedialog.askopenfilename(title="Select Image for Interactive Mode", filetypes=filetypes)
                if not path:
                    self._log("Interactive mode cancelled - no image selected")
                    return None
                self.input_path.set(path)
                input_path = path
                self._log(f"Interactive mode - selected: {path}")
        elif not input_path:
            messagebox.showerror("Error", "Please select an input!")
            return None
        
        # 构建命令
        exe_path = get_executable_path()
        
        if is_frozen_app():
            # 打包后：使用 main_launcher 或直接调用内部 Python
            cmd = [str(exe_path), model]
        else:
            # 开发环境：使用 Python 运行 main_launcher
            launcher = RESOURCE_PATH / "main_launcher.py"
            cmd = [str(exe_path), str(launcher), model]
        
        if mode == "interactive":
            cmd.extend(["--interactive", "--input", input_path])
        else:
            cmd.extend(["--input", input_path])
            if mode == "batch":
                cmd.append("--batch")
        
        return cmd
    
    def _run(self):
        """Run segmentation."""
        if self.is_running:
            return
        
        model = self.model_var.get()
        mode = self.mode_var.get()
        input_path = self.input_path.get()
        
        # Interactive 模式：使用嵌入式 GUI
        if mode == "interactive":
            self._run_interactive_embedded(model, input_path)
            return
        
        # Auto/Batch 模式：使用子进程
        cmd = self._build_command()
        if not cmd:
            return
        
        self.is_running = True
        self.run_btn.config(state=tk.DISABLED)
        self.status_var.set("Running...")
        self._log(f"Starting {model}...")
        self._log(f"Command: {' '.join(cmd)}")
        
        thread = threading.Thread(target=self._run_process, args=(cmd,))
        thread.daemon = True
        thread.start()
    
    def _run_interactive_embedded(self, model, input_path):
        """运行嵌入式 Interactive 模式（直接在 GUI 中）"""
        try:
            # 检查是否需要选择图片
            if not input_path:
                filetypes = [("Images", "*.tif *.tiff *.jpg *.jpeg *.png *.bmp")]
                path = filedialog.askopenfilename(title="Select Image for Interactive Mode", filetypes=filetypes)
                if not path:
                    self._log("Interactive mode cancelled - no image selected")
                    return
                input_path = path
                self.input_path.set(path)
            
            self._log("=" * 50)
            self._log(f"Starting Embedded Interactive Mode ({model.upper()})")
            self._log("=" * 50)
            self._log(f"Image: {input_path}")
            
            # 导入并启动嵌入式 interactive
            try:
                from gui_interactive import run_interactive_gui
                run_interactive_gui(self.root, model, input_path)
                self._log("✓ Interactive window opened")
            except ImportError as e:
                self._log(f"✗ Cannot load interactive module: {e}")
                messagebox.showerror("Error", f"Failed to start interactive mode: {e}")
                
        except Exception as e:
            self._log(f"✗ Error: {e}")
            messagebox.showerror("Error", f"Interactive mode error: {e}")
    
    def _run_process(self, cmd):
        """Run process in background (for Auto/Batch modes only)."""
        try:
            # 设置工作目录
            cwd = str(RESOURCE_PATH) if is_frozen_app() else None
            
            env = os.environ.copy()
            env['PYTHONPATH'] = str(RESOURCE_PATH)
            
            # 普通模式（Auto/Batch）：正常捕获输出
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, bufsize=1, cwd=cwd, env=env
            )
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    self.root.after(0, lambda l=line: self._log(l.strip()))
            
            self.process.wait()
            if self.process.returncode == 0:
                self.root.after(0, self._on_success)
            else:
                self.root.after(0, lambda: self._on_error(f"Exit code: {self.process.returncode}"))
                    
        except Exception as e:
            self.root.after(0, lambda: self._on_error(str(e)))
        finally:
            self.is_running = False
            self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL))
    
    def _on_success(self):
        self.status_var.set("Completed")
        self._log("✓ Done!")
        messagebox.showinfo("Success", "Segmentation completed!")
    
    def _on_error(self, msg=None):
        self.status_var.set("Error")
        self._log(f"✗ Failed: {msg or 'Unknown error'}")
    
    def _stop(self):
        """Stop process."""
        if self.process:
            self.process.terminate()
            self._log("Stopped")
            self.status_var.set("Stopped")
    
    def _open_results(self):
        """Open results folder."""
        results_paths = [
            Path("results").absolute(),
            RESOURCE_PATH / "results",
            RESOURCE_PATH.parent / "results",
        ]
        
        results_path = results_paths[0]
        for path in results_paths:
            if path.exists():
                results_path = path
                break
        
        results_path.mkdir(exist_ok=True)
        if sys.platform == "darwin":
            subprocess.run(["open", results_path])
        elif sys.platform == "win32":
            os.startfile(results_path)
        else:
            subprocess.run(["xdg-open", results_path])


def run_as_subprocess():
    """作为子进程运行 - 执行实际的模型处理"""
    # 这个函数在打包后的应用中被子进程调用时使用
    import argparse
    
    parser = argparse.ArgumentParser(description='FastMeasure Subprocess')
    parser.add_argument('model', choices=['fastsam', 'mobilesam'], help='Model to use')
    parser.add_argument('--input', '-i', required=True, help='Input image path')
    parser.add_argument('--batch', '-b', action='store_true', help='Batch mode')
    parser.add_argument('--interactive', '-t', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    # 设置资源路径
    if is_frozen_app():
        bundle_dir = Path(sys._MEIPASS)
    else:
        bundle_dir = Path(__file__).parent
    sys.path.insert(0, str(bundle_dir))
    
    # 导入并运行 main_launcher 的功能
    try:
        # 动态导入 main_launcher 的函数
        import importlib.util
        launcher_path = bundle_dir / "main_launcher.py"
        
        if launcher_path.exists():
            spec = importlib.util.spec_from_file_location("main_launcher", launcher_path)
            launcher = importlib.util.module_from_spec(spec)
            sys.modules["main_launcher"] = launcher
            spec.loader.exec_module(launcher)
            
            if args.interactive:
                return launcher.run_interactive(args.model, args.input)
            elif args.model == 'fastsam':
                return launcher.run_fastsam_auto(args.input, args.batch)
            else:
                return launcher.run_mobilesam_auto(args.input, args.batch)
        else:
            print(f"Error: main_launcher.py not found at {launcher_path}")
            return 1
            
    except Exception as e:
        print(f"Error in subprocess: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    # 检查是否作为子进程运行（有命令行参数 model）
    if len(sys.argv) > 1 and sys.argv[1] in ['fastsam', 'mobilesam']:
        # 子进程模式 - 执行实际处理
        sys.exit(run_as_subprocess())
    
    # 主 GUI 模式
    root = tk.Tk()
    app = FastMeasureSimpleGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
