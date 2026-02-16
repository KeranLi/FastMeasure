"""
Model Manager for FastMeasure

Handles automatic model download, verification, and management.
Simplifies model setup for end users.
"""

import os
import hashlib
import urllib.request
from pathlib import Path
from typing import Optional, Dict, Callable
import tkinter as tk
from tkinter import ttk, messagebox


# Model definitions with download URLs and checksums
MODELS = {
    "yolo": {
        "name": "YOLO Detection Model",
        "filename": "best_yolo_20260107.pt",
        "description": "Main detection model for grain localization",
        "size_mb": 100,
        "url": "https://github.com/KeranLi/FastMeasure/releases/download/v1.0.0/best_yolo_20260107.pt",
        "md5": None,  # Add actual MD5 when available
    },
    "fastsam": {
        "name": "FastSAM Model",
        "filename": "FastSAM-s.pt",
        "description": "Fast segmentation model (recommended for speed)",
        "size_mb": 150,
        "url": "https://github.com/KeranLi/FastMeasure/releases/download/v1.0.0/FastSAM-s.pt",
        "md5": None,
    },
    "mobilesam": {
        "name": "MobileSAM Model",
        "filename": "mobile_sam.pt",
        "description": "High-quality segmentation model (slower but more accurate)",
        "size_mb": 450,
        "url": "https://github.com/KeranLi/FastMeasure/releases/download/v1.0.0/mobile_sam.pt",
        "md5": None,
    }
}


class ModelManager:
    """Manages model files for FastMeasure."""
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize model manager.
        
        Args:
            models_dir: Directory to store model files
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
    
    def check_model(self, model_key: str) -> bool:
        """
        Check if a model file exists.
        
        Args:
            model_key: Key from MODELS dict (yolo, fastsam, mobilesam)
            
        Returns:
            True if model exists
        """
        if model_key not in MODELS:
            return False
        
        model_info = MODELS[model_key]
        model_path = self.models_dir / model_info["filename"]
        
        return model_path.exists() and model_path.stat().st_size > 0
    
    def check_all_models(self) -> Dict[str, bool]:
        """Check status of all models."""
        return {key: self.check_model(key) for key in MODELS.keys()}
    
    def get_model_path(self, model_key: str) -> Optional[Path]:
        """
        Get path to model file if it exists.
        
        Args:
            model_key: Key from MODELS dict
            
        Returns:
            Path to model or None if not found
        """
        if self.check_model(model_key):
            return self.models_dir / MODELS[model_key]["filename"]
        return None
    
    def download_model(
        self,
        model_key: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        chunk_size: int = 8192
    ) -> bool:
        """
        Download a model file.
        
        Args:
            model_key: Key from MODELS dict
            progress_callback: Function(current_bytes, total_bytes) for progress updates
            chunk_size: Download chunk size in bytes
            
        Returns:
            True if download successful
        """
        if model_key not in MODELS:
            print(f"[ModelManager] Unknown model: {model_key}")
            return False
        
        model_info = MODELS[model_key]
        url = model_info.get("url")
        
        if not url:
            print(f"[ModelManager] No download URL for {model_key}")
            return False
        
        output_path = self.models_dir / model_info["filename"]
        
        try:
            print(f"[ModelManager] Downloading {model_info['name']}...")
            print(f"[ModelManager] URL: {url}")
            print(f"[ModelManager] Output: {output_path}")
            
            # Open connection
            req = urllib.request.Request(url, headers={'User-Agent': 'FastMeasure/1.0'})
            
            with urllib.request.urlopen(req, timeout=30) as response:
                total_size = int(response.headers.get('Content-Length', 0))
                downloaded = 0
                
                with open(output_path, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress_callback and total_size > 0:
                            progress_callback(downloaded, total_size)
            
            # Verify file size
            actual_size = output_path.stat().st_size
            expected_size = model_info["size_mb"] * 1024 * 1024
            
            if actual_size < expected_size * 0.9:  # Allow 10% variance
                print(f"[ModelManager] Warning: Downloaded file seems too small ({actual_size} bytes)")
                return False
            
            print(f"[ModelManager] Download completed: {output_path}")
            return True
            
        except Exception as e:
            print(f"[ModelManager] Download failed: {e}")
            # Clean up partial download
            if output_path.exists():
                output_path.unlink()
            return False
    
    def get_missing_models(self) -> list:
        """Get list of missing model keys."""
        return [key for key in MODELS.keys() if not self.check_model(key)]
    
    def get_model_info(self, model_key: str) -> Optional[Dict]:
        """Get information about a model."""
        return MODELS.get(model_key)


class ModelDownloadDialog:
    """GUI dialog for downloading models."""
    
    def __init__(self, parent: tk.Tk, model_manager: ModelManager):
        """
        Initialize download dialog.
        
        Args:
            parent: Parent Tk window
            model_manager: ModelManager instance
        """
        self.parent = parent
        self.model_manager = model_manager
        self.dialog = None
        self.downloading = False
    
    def show(self):
        """Show the download dialog."""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("Model Manager")
        self.dialog.geometry("500x400")
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        # Title
        ttk.Label(
            self.dialog,
            text="Model Download Manager",
            font=("Helvetica", 14, "bold")
        ).pack(pady=10)
        
        # Model list frame
        list_frame = ttk.Frame(self.dialog, padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        
        self.model_vars = {}
        self.status_labels = {}
        
        for model_key, model_info in MODELS.items():
            frame = ttk.Frame(list_frame)
            frame.pack(fill=tk.X, pady=5)
            
            # Checkbox
            var = tk.BooleanVar(value=not self.model_manager.check_model(model_key))
            self.model_vars[model_key] = var
            
            cb = ttk.Checkbutton(
                frame,
                text=f"{model_info['name']} ({model_info['size_mb']} MB)",
                variable=var
            )
            cb.pack(side=tk.LEFT)
            
            if self.model_manager.check_model(model_key):
                cb.config(state=tk.DISABLED)
                status_text = "✓ Installed"
            else:
                status_text = "✗ Not installed"
            
            status_label = ttk.Label(frame, text=status_text)
            status_label.pack(side=tk.RIGHT)
            self.status_labels[model_key] = status_label
            
            # Description
            ttk.Label(
                list_frame,
                text=f"  {model_info['description']}",
                font=("Helvetica", 8),
                foreground="gray"
            ).pack(anchor=tk.W)
        
        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            self.dialog,
            variable=self.progress_var,
            maximum=100,
            mode='determinate'
        )
        self.progress_bar.pack(fill=tk.X, padx=20, pady=10)
        
        self.status_label = ttk.Label(self.dialog, text="Ready")
        self.status_label.pack()
        
        # Buttons
        btn_frame = ttk.Frame(self.dialog)
        btn_frame.pack(pady=10)
        
        self.download_btn = ttk.Button(
            btn_frame,
            text="Download Selected",
            command=self._start_download
        )
        self.download_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            btn_frame,
            text="Close",
            command=self.dialog.destroy
        ).pack(side=tk.LEFT, padx=5)
    
    def _start_download(self):
        """Start downloading selected models."""
        if self.downloading:
            return
        
        # Get selected models
        to_download = [
            key for key, var in self.model_vars.items()
            if var.get() and not self.model_manager.check_model(key)
        ]
        
        if not to_download:
            messagebox.showinfo("Info", "No models selected for download")
            return
        
        self.downloading = True
        self.download_btn.config(state=tk.DISABLED)
        
        # Download in thread
        import threading
        thread = threading.Thread(target=self._download_thread, args=(to_download,))
        thread.daemon = True
        thread.start()
    
    def _download_thread(self, model_keys: list):
        """Download models in background thread."""
        for model_key in model_keys:
            self.dialog.after(0, lambda k=model_key: self._update_status(f"Downloading {MODELS[k]['name']}..."))
            
            def progress(current, total):
                percent = (current / total) * 100 if total > 0 else 0
                self.dialog.after(0, lambda p=percent: self.progress_var.set(p))
            
            success = self.model_manager.download_model(model_key, progress_callback=progress)
            
            if success:
                self.dialog.after(0, lambda k=model_key: self._mark_installed(k))
            else:
                self.dialog.after(0, lambda k=model_key: self._mark_failed(k))
        
        self.dialog.after(0, self._download_complete)
    
    def _update_status(self, text: str):
        """Update status label."""
        self.status_label.config(text=text)
    
    def _mark_installed(self, model_key: str):
        """Mark model as installed."""
        self.status_labels[model_key].config(text="✓ Installed", foreground="green")
        self.model_vars[model_key].set(False)
    
    def _mark_failed(self, model_key: str):
        """Mark model as failed."""
        self.status_labels[model_key].config(text="✗ Failed", foreground="red")
    
    def _download_complete(self):
        """Handle download completion."""
        self.downloading = False
        self.download_btn.config(state=tk.NORMAL)
        self.progress_var.set(100)
        self.status_label.config(text="Download complete")
        messagebox.showinfo("Complete", "Model download completed!")


def check_and_download_models(parent: tk.Tk, models_dir: str = "models") -> bool:
    """
    Check if required models are present, prompt to download if missing.
    
    Args:
        parent: Parent Tk window
        models_dir: Models directory path
        
    Returns:
        True if all required models are available
    """
    manager = ModelManager(models_dir)
    
    # Check for YOLO model (required)
    if not manager.check_model("yolo"):
        response = messagebox.askyesno(
            "Models Required",
            "YOLO detection model is required but not found.\n\n"
            "Would you like to download it now?\n"
            "(Requires internet connection)"
        )
        
        if response:
            dialog = ModelDownloadDialog(parent, manager)
            dialog.show()
            parent.wait_window(dialog.dialog)
            
            # Check again
            if not manager.check_model("yolo"):
                messagebox.showerror(
                    "Error",
                    "YOLO model is required to run FastMeasure.\n"
                    "Please download models manually from:\n"
                    "https://github.com/KeranLi/FastMeasure/releases"
                )
                return False
        else:
            return False
    
    return True


if __name__ == "__main__":
    # Test model manager
    manager = ModelManager()
    
    print("Model Status:")
    for key, exists in manager.check_all_models().items():
        status = "✓" if exists else "✗"
        print(f"  {status} {MODELS[key]['name']}")
    
    print(f"\nMissing models: {manager.get_missing_models()}")
