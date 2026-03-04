#!/usr/bin/env python
"""
FastMeasure GUI Launcher - Standalone Desktop Application

A user-friendly graphical interface for running FastMeasure without command line.
Designed for end users who prefer GUI over CLI.

Features:
- One-click image/folder selection
- Mode selection (FastSAM/MobileSAM, Auto/Batch/Interactive)
- Real-time progress display
- Result viewing
- Model management
"""

import sys
import os

# Fix for PyInstaller + OpenCV on macOS
# Must be set before importing cv2
if sys.platform == "darwin":
    os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"
    # Prevent cv2 recursion in frozen app
    if getattr(sys, 'frozen', False):
        # Running in a bundle
        bundle_dir = sys._MEIPASS
        os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = '1000000000'

import threading
import subprocess
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext


def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and PyInstaller"""
    # Try multiple possible locations
    possible_paths = []
    
    if getattr(sys, 'frozen', False):
        # Running in PyInstaller bundle
        # sys._MEIPASS is the temp extraction directory (_internal folder)
        possible_paths.append(Path(sys._MEIPASS) / relative_path)
        # Also try the executable's directory
        possible_paths.append(Path(sys.executable).parent / relative_path)
    else:
        # Running in normal Python environment
        possible_paths.append(Path(__file__).parent.parent / relative_path)
    
    # Try current working directory (for compatibility)
    possible_paths.append(Path(relative_path))
    
    # Try executable's parent directory
    possible_paths.append(Path(sys.executable).parent / relative_path)
    
    # Find the first existing path
    for path in possible_paths:
        if path.exists():
            return path
    
    # If not found, return the first option (will fail with proper error message)
    return possible_paths[0]


def ensure_configs_available():
    """Ensure config files are available in the working directory.
    In PyInstaller, copy configs from _internal to working directory if needed."""
    configs_dir = Path("configs")
    
    # If configs already exists in working directory, use it
    if configs_dir.exists() and (configs_dir / "geometry.yaml").exists():
        return configs_dir
    
    # Try to find configs in PyInstaller bundle
    if getattr(sys, 'frozen', False):
        source_configs = Path(sys._MEIPASS) / "configs"
        if source_configs.exists():
            # Copy configs to working directory
            import shutil
            configs_dir.mkdir(exist_ok=True)
            for config_file in source_configs.glob("*.yaml"):
                dest = configs_dir / config_file.name
                if not dest.exists():
                    shutil.copy2(config_file, dest)
                    print(f"Copied config: {config_file.name}")
            return configs_dir
    
    return configs_dir


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.seg_tools import FileUtils


class FastMeasureGUI:
    """Main GUI application for FastMeasure."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("FastMeasure - Rock Grain Segmentation")
        self.root.geometry("800x700")
        self.root.minsize(700, 600)
        
        # Ensure configs are available (important for PyInstaller)
        ensure_configs_available()
        
        # Set icon if available
        try:
            self.root.iconbitmap("assets/icon.ico")
        except:
            pass
        
        # Variables
        self.input_path = tk.StringVar()
        self.model_var = tk.StringVar(value="fastsam")
        self.mode_var = tk.StringVar(value="auto")
        self.device_var = tk.StringVar(value="cpu")
        self.status_var = tk.StringVar(value="Ready")
        self.progress_var = tk.DoubleVar(value=0)
        
        self.process = None
        self.is_running = False
        
        # Build UI
        self._create_menu()
        self._create_main_frame()
        self._create_status_bar()
        
        # Check environment
        self._check_setup()
    
    def _create_menu(self):
        """Create application menu."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Select Image...", command=self._select_image)
        file_menu.add_command(label="Select Folder...", command=self._select_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Open Results Folder", command=self._open_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Download Models", command=self._download_models)
        tools_menu.add_command(label="Check System", command=self._check_system)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Quick Start", command=self._show_help)
        help_menu.add_command(label="About", command=self._show_about)
    
    def _create_main_frame(self):
        """Create main application frame."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="FastMeasure",
            font=("Helvetica", 20, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 5))
        
        subtitle_label = ttk.Label(
            main_frame,
            text="Rock Grain Auto Segmentation System",
            font=("Helvetica", 10)
        )
        subtitle_label.grid(row=1, column=0, columnspan=3, pady=(0, 20))
        
        # Input Section
        input_frame = ttk.LabelFrame(main_frame, text="Input", padding="10")
        input_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        input_frame.columnconfigure(1, weight=1)
        
        ttk.Label(input_frame, text="Path:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(input_frame, textvariable=self.input_path).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        
        btn_frame = ttk.Frame(input_frame)
        btn_frame.grid(row=0, column=2)
        ttk.Button(btn_frame, text="Image", command=self._select_image, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Folder", command=self._select_folder, width=8).pack(side=tk.LEFT, padx=2)
        
        # Settings Section
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Model Selection
        ttk.Label(settings_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, padx=5)
        model_combo = ttk.Combobox(
            settings_frame, 
            textvariable=self.model_var,
            values=["fastsam", "mobilesam"],
            state="readonly",
            width=15
        )
        model_combo.grid(row=0, column=1, sticky=tk.W, padx=5)
        model_combo.bind("<<ComboboxSelected>>", self._on_model_change)
        
        ttk.Label(settings_frame, text="Fast: FastSAM, Precise: MobileSAM").grid(
            row=0, column=2, sticky=tk.W, padx=5
        )
        
        # Mode Selection
        ttk.Label(settings_frame, text="Mode:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        mode_frame = ttk.Frame(settings_frame)
        mode_frame.grid(row=1, column=1, columnspan=2, sticky=tk.W, padx=5)
        
        ttk.Radiobutton(
            mode_frame, text="Auto (Single)", variable=self.mode_var, value="auto"
        ).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(
            mode_frame, text="Batch (Folder)", variable=self.mode_var, value="batch"
        ).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(
            mode_frame, text="Interactive", variable=self.mode_var, value="interactive"
        ).pack(side=tk.LEFT, padx=5)
        
        # Device Selection
        ttk.Label(settings_frame, text="Device:").grid(row=2, column=0, sticky=tk.W, padx=5)
        device_frame = ttk.Frame(settings_frame)
        device_frame.grid(row=2, column=1, columnspan=2, sticky=tk.W, padx=5)
        
        ttk.Radiobutton(
            device_frame, text="CPU", variable=self.device_var, value="cpu"
        ).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(
            device_frame, text="CUDA (GPU)", variable=self.device_var, value="cuda"
        ).pack(side=tk.LEFT, padx=5)
        
        # Action Buttons
        action_frame = ttk.Frame(main_frame)
        action_frame.grid(row=4, column=0, columnspan=3, pady=15)
        
        self.run_btn = ttk.Button(
            action_frame, 
            text="▶ Run Segmentation", 
            command=self._run,
            width=20
        )
        self.run_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            action_frame, 
            text="⏹ Stop", 
            command=self._stop,
            width=10
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            action_frame, 
            text="📁 Open Results", 
            command=self._open_results,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        # Progress Bar
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_bar = ttk.Progressbar(
            progress_frame, 
            variable=self.progress_var, 
            maximum=100,
            mode='determinate'
        )
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5)
        
        # Log Output
        log_frame = ttk.LabelFrame(main_frame, text="Output Log", padding="5")
        log_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(6, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame, 
            wrap=tk.WORD, 
            height=15,
            font=("Consolas", 9)
        )
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Initial message
        self._log("FastMeasure GUI Started")
        self._log("Select an image or folder to begin")
    
    def _create_status_bar(self):
        """Create status bar at bottom."""
        status_bar = ttk.Frame(self.root, relief=tk.SUNKEN, padding=(5, 2))
        status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        ttk.Label(status_bar, textvariable=self.status_var).pack(side=tk.LEFT)
        ttk.Separator(status_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Label(status_bar, text="CPU Mode Ready").pack(side=tk.LEFT)
    
    def _check_setup(self):
        """Check if models and configs are present."""
        models_dir = Path("models")
        if not models_dir.exists() or not list(models_dir.glob("*.pt")):
            self._log("⚠ Warning: No model files found in ./models/")
            self._log("Please download models using Tools > Download Models")
    
    def _select_image(self):
        """Select input image file."""
        filetypes = [
            ("Image files", "*.tif *.tiff *.jpg *.jpeg *.png *.bmp"),
            ("All files", "*.*")
        ]
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=filetypes
        )
        if path:
            self.input_path.set(path)
            self.mode_var.set("auto")
            self._log(f"Selected image: {path}")
    
    def _select_folder(self):
        """Select input folder."""
        path = filedialog.askdirectory(title="Select Folder")
        if path:
            self.input_path.set(path)
            self.mode_var.set("batch")
            self._log(f"Selected folder: {path}")
    
    def _on_model_change(self, event=None):
        """Handle model selection change."""
        model = self.model_var.get()
        self._log(f"Selected model: {model}")
    
    def _run(self):
        """Run segmentation."""
        if self.is_running:
            messagebox.showwarning("Warning", "A process is already running!")
            return
        
        # Get parameters
        model = self.model_var.get()
        mode = self.mode_var.get()
        device = self.device_var.get()
        input_path = self.input_path.get()
        
        # Validate
        if mode != "interactive" and not input_path:
            messagebox.showerror("Error", "Please select an input image or folder!")
            return
        
        if mode != "interactive" and not Path(input_path).exists():
            messagebox.showerror("Error", "Selected path does not exist!")
            return
        
        # Run in thread
        self.is_running = True
        self.run_btn.config(state=tk.DISABLED)
        self.status_var.set("Running...")
        self.progress_var.set(0)
        self._log("=" * 50)
        self._log(f"Starting {model.upper()} segmentation...")
        self._log(f"Mode: {mode}, Device: {device}")
        
        thread = threading.Thread(target=self._run_direct, args=(model, mode, device, input_path))
        thread.daemon = True
        thread.start()
    
    def _run_direct(self, model, mode, device, input_path):
        """Run segmentation directly (works in both dev and frozen environments)."""
        import io
        import contextlib
        
        try:
            # Redirect stdout to capture output
            output_buffer = io.StringIO()
            
            with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(output_buffer):
                if model == "fastsam":
                    result = self._run_fastsam(mode, device, input_path)
                else:
                    result = self._run_mobilesam(mode, device, input_path)
            
            # Get captured output
            output = output_buffer.getvalue()
            
            # Log output line by line
            for line in output.split('\n'):
                if line.strip():
                    self.root.after(0, lambda l=line: self._log(l))
            
            if result == 0:
                self.root.after(0, self._on_success)
            else:
                self.root.after(0, lambda: self._on_error(f"Process returned {result}"))
                
        except Exception as e:
            self.root.after(0, lambda: self._on_error(str(e)))
            import traceback
            traceback.print_exc()
        finally:
            self.is_running = False
            self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL))
    
    def _run_fastsam(self, mode, device, input_path):
        """Run FastSAM segmentation (matching run_fastsam.py functionality)."""
        from fastsam.rock_fastsam_system import RockUltraSystem
        from core import update_config_from_args, print_summary
        
        # Ensure configs are available and get configs directory
        configs_dir = ensure_configs_available()
        
        # Set device in config
        import yaml
        config_path = configs_dir / "fastsam.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        config['model_paths']['device'] = device
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Create system instance
        try:
            system = RockUltraSystem(str(config_path))
        except Exception as e:
            print(f"Error: System initialization failed: {e}")
            return 1
        
        # Initialize models
        print("\nInitializing AI models...")
        if not system.initialize_models():
            print("Error: Model initialization failed")
            return 1
        print("AI models initialized successfully")
        
        # Show system info
        system.show_system_info()
        
        # Interactive mode
        if mode == "interactive":
            if not input_path:
                print("Error: Interactive mode requires input image")
                return 1
            
            print("\nStarting GUI interactive mode...")
            system.run_interactive_mode(input_path)
            
            print("\n" + "=" * 60)
            print("GUI interaction ended")
            print("=" * 60)
            
            # Save interactive results (matching run_fastsam.py)
            if system.interactive_system and hasattr(system.interactive_system, 'grains'):
                grains_count = len(system.interactive_system.grains)
                if grains_count > 0:
                    print(f"Marked {grains_count} grains during interaction")
                    # Note: In GUI mode, auto-save instead of prompting
                    if hasattr(system.interactive_system, '_generate_complete_outputs'):
                        output_dir = system.interactive_system._generate_complete_outputs()
                        if output_dir:
                            print(f"Results saved to: {output_dir}")
            
            return 0
        
        # Check input path
        if not input_path:
            print("Error: No input path specified")
            return 1
        
        if not Path(input_path).exists():
            print(f"Error: Input path does not exist: {input_path}")
            return 1
        
        # Process
        results = None
        
        if Path(input_path).is_file():
            print(f"\nProcessing single image: {input_path}")
            results = system.process_single_image(input_path)
            
            # Check scale detection status
            if results and results.get('scale_detection_success'):
                print(f"✓ Scale detection successful: {results.get('scale_factor', 'N/A')} μm/px")
                print(f"  - area_um2 and diameter_um calculated")
            else:
                print("⚠ Scale detection failed or disabled")
                print("  - area_um2 and diameter_um not available")
                print("  - Check if image has red scale bar at bottom-right corner")
        elif Path(input_path).is_dir():
            if mode == "batch":
                print(f"\nBatch processing folder: {input_path}")
                results = system.batch_process(input_path)
            else:
                print("Error: Input path is folder, please use batch mode")
                return 1
        
        # Show results summary (matching run_fastsam.py)
        if results:
            print_summary(results, "FastSAM")
        
        print(f"\nAll results saved to: {system.output_root}")
        print("\n" + "=" * 70)
        print("FastSAM processing complete!")
        print("=" * 70)
        
        return 0
    
    def _run_mobilesam(self, mode, device, input_path):
        """Run MobileSAM segmentation (matching run_mobilesam.py functionality)."""
        try:
            from mobilesam.rock_mobilesam_system import RockMobileSystem
        except ImportError:
            print("Error: MobileSAM not installed")
            print("Install with: pip install git+https://github.com/ChaoningZhang/MobileSAM.git")
            return 1
        
        from core import print_summary
        
        # Ensure configs are available and get configs directory
        configs_dir = ensure_configs_available()
        
        # Set device in config
        import yaml
        config_path = configs_dir / "mobilesam.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        config['model_paths']['device'] = device
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Create system instance
        try:
            print(f"\nInitializing MobileSAM main system...")
            system = RockMobileSystem(str(config_path))
        except Exception as e:
            print(f"Error: System initialization failed: {e}")
            return 1
        
        # Initialize models
        print("Initializing AI models...")
        if not system.initialize_models():
            print("Error: Model initialization failed")
            return 1
        print("AI models initialized successfully")
        
        # Show system info
        system.show_system_info()
        
        # Interactive mode
        if mode == "interactive":
            if not input_path:
                print("Error: Interactive mode requires input image")
                return 1
            
            print(f"\nStarting GUI interactive mode...")
            try:
                system.run_interactive_mode(input_path)
                
                print("\n" + "=" * 60)
                print("GUI interaction ended")
                print("=" * 60)
                
                # Save interactive results (matching run_mobilesam.py)
                if system.interactive_system and hasattr(system.interactive_system, 'grains'):
                    grains_count = len(system.interactive_system.grains)
                    if grains_count > 0:
                        print(f"Marked {grains_count} grains during interaction")
                        if hasattr(system.interactive_system, '_generate_complete_outputs'):
                            output_dir = system.interactive_system._generate_complete_outputs()
                            if output_dir:
                                print(f"Results saved to: {output_dir}")
                
                return 0
            except Exception as e:
                print(f"Error: Interactive mode execution failed: {e}")
                return 1
        
        # Check input
        if not input_path:
            print("\nError: No input path specified")
            return 1
        
        input_path = Path(input_path).absolute()
        
        if not input_path.exists():
            print(f"Error: Input path does not exist: {input_path}")
            return 1
        
        # Process
        if input_path.is_file():
            print(f"\nProcessing single image: {input_path}")
            results = system.process_single_image(str(input_path))
            
            # Check scale detection status
            if results and results.get('scale_detection_success'):
                print(f"✓ Scale detection successful: {results.get('scale_factor', 'N/A')} μm/px")
                print(f"  - area_um2 and diameter_um calculated")
            else:
                print("⚠ Scale detection failed or disabled")
                print("  - area_um2 and diameter_um not available")
                print("  - Check if image has red scale bar at bottom-right corner")
            
            if results:
                print_summary(results, "MobileSAM")
        elif input_path.is_dir():
            if mode == "batch":
                print(f"\nBatch processing folder: {input_path}")
                results = system.batch_process(str(input_path))
                if results:
                    print_summary(results, "MobileSAM")
            else:
                print("Error: Input path is folder, please use batch mode")
                return 1
        
        print(f"\nAll results saved to: {system.output_root}")
        print("\n" + "=" * 70)
        print("MobileSAM processing complete!")
        print("=" * 70)
        
        return 0
    
    def _on_success(self):
        """Handle successful completion."""
        self.status_var.set("Completed")
        self.progress_var.set(100)
        self._log("=" * 50)
        self._log("✓ Segmentation completed successfully!")
        self._log("Results saved to ./results/")
        
        if messagebox.askyesno("Success", "Segmentation completed!\nOpen results folder?"):
            self._open_results()
    
    def _on_error(self, error_msg=None):
        """Handle error."""
        self.status_var.set("Error")
        self._log("=" * 50)
        self._log("✗ Segmentation failed!")
        if error_msg:
            self._log(f"Error: {error_msg}")
        
        messagebox.showerror("Error", f"Segmentation failed!\n{error_msg or ''}")
    
    def _stop(self):
        """Stop running process."""
        if self.process and self.is_running:
            self.process.terminate()
            self._log("Process terminated by user")
            self.status_var.set("Stopped")
            self.is_running = False
            self.run_btn.config(state=tk.NORMAL)
    
    def _open_results(self):
        """Open results folder."""
        results_path = Path("results").absolute()
        results_path.mkdir(exist_ok=True)
        
        if sys.platform == "win32":
            os.startfile(results_path)
        elif sys.platform == "darwin":
            subprocess.run(["open", results_path])
        else:
            subprocess.run(["xdg-open", results_path])
    
    def _download_models(self):
        """Open model download dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Download Models")
        dialog.geometry("500x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Model Download", font=("Helvetica", 14, "bold")).pack(pady=10)
        
        # Model list
        models_frame = ttk.Frame(dialog, padding="10")
        models_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        
        models = [
            ("YOLO Detection Model", "best_yolo_20260107.pt", "~100 MB"),
            ("FastSAM Model", "FastSAM-s.pt", "~150 MB"),
            ("MobileSAM Model", "mobile_sam.pt", "~450 MB"),
        ]
        
        for name, filename, size in models:
            frame = ttk.Frame(models_frame)
            frame.pack(fill=tk.X, pady=5)
            
            exists = "✓" if (Path("models") / filename).exists() else "✗"
            status_color = "green" if exists == "✓" else "red"
            
            ttk.Label(frame, text=f"{exists} {name}", font=("Helvetica", 10)).pack(side=tk.LEFT)
            ttk.Label(frame, text=f"({size})").pack(side=tk.RIGHT)
        
        # Instructions
        ttk.Label(
            dialog, 
            text="Please manually download models and place them in the 'models' folder:",
            wraplength=450
        ).pack(pady=10)
        
        ttk.Label(
            dialog,
            text="https://github.com/KeranLi/FastMeasure/releases",
            foreground="blue",
            cursor="hand2"
        ).pack()
        
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)
    
    def _check_system(self):
        """Check system requirements."""
        dialog = tk.Toplevel(self.root)
        dialog.title("System Check")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        
        text = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, padding="10")
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Run checks
        checks = []
        
        # Python version
        checks.append(f"Python: {sys.version}")
        
        # PyTorch
        try:
            import torch
            checks.append(f"PyTorch: {torch.__version__}")
            checks.append(f"CUDA Available: {torch.cuda.is_available()}")
        except:
            checks.append("PyTorch: Not installed!")
        
        # Other packages
        packages = ["cv2", "numpy", "pandas", "ultralytics"]
        for pkg in packages:
            try:
                __import__(pkg)
                checks.append(f"{pkg}: OK")
            except:
                checks.append(f"{pkg}: Missing!")
        
        text.insert(tk.END, "\n".join(checks))
        text.config(state=tk.DISABLED)
        
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)
    
    def _show_help(self):
        """Show help dialog."""
        help_text = """
FastMeasure Quick Start:

1. Select Input:
   - Click 'Image' to select a single image
   - Click 'Folder' to batch process a folder

2. Choose Settings:
   - Model: FastSAM (fast) or MobileSAM (precise)
   - Mode: Auto, Batch, or Interactive
   - Device: CPU or CUDA (GPU)

3. Run:
   - Click 'Run Segmentation' to start
   - Progress will be shown in the log

4. View Results:
   - Results are saved in ./results/
   - Click 'Open Results' to view

Interactive Mode:
   - Left click: Add foreground point
   - Right click: Add background point
   - S: Save results
   - X: Delete last grain
   - Q: Quit
        """
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Quick Start")
        dialog.geometry("500x500")
        dialog.transient(self.root)
        
        text = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, padding="10", font=("Helvetica", 10))
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text.insert(tk.END, help_text)
        text.config(state=tk.DISABLED)
        
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)
    
    def _show_about(self):
        """Show about dialog."""
        messagebox.showinfo(
            "About FastMeasure",
            """FastMeasure - Rock Grain Auto Segmentation System

Version: 1.0.0

A professional tool for processing rock microscopic images,
automatically detecting and segmenting grains.

Inspired by segmenteverygrain by Zoltán Sylvester.

Features:
• YOLO + FastSAM/MobileSAM
• Automatic scale bar detection
• 10+ geometric parameters
• Interactive segmentation

https://github.com/KeranLi/FastMeasure
            """
        )
    
    def _log(self, message):
        """Add message to log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)


def main():
    """Main entry point."""
    # Create root window
    root = tk.Tk()
    
    # Set DPI awareness on Windows
    if sys.platform == "win32":
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except:
            pass
    
    # Create and run app
    app = FastMeasureGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
