import sys
import os
import json
import cv2
import numpy as np
import torch
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from huggingface_hub import hf_hub_download

# ================= CONFIG & PATHS =================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path: sys.path.append(ROOT_DIR)

DATASET_DIR = None
possible_paths = [
    os.path.join(os.path.dirname(ROOT_DIR), "Dataset"), 
    os.path.join(ROOT_DIR, "Dataset"),
    r"hybrid-cell-classification\Dataset"
]
for path in possible_paths:
    if os.path.exists(path) and os.path.exists(os.path.join(path, "images")):
        DATASET_DIR = path
        break

if not DATASET_DIR: DATASET_DIR = os.path.join(ROOT_DIR, "Dataset")

IMG_DIR = os.path.join(DATASET_DIR, "images")
CNT_DIR = os.path.join(DATASET_DIR, "contour")
OUT_DIR = os.path.join(DATASET_DIR, "output", "final_visualization")
JSON_OUT_DIR = os.path.join(DATASET_DIR, "output", "json_results")

HF_ID = "SoftmaxSamurai/ConvNext_SCTC"
HF_FILE = "best_model.pth"
CACHE_DIR = os.path.join(ROOT_DIR, "weights")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

WEIGHTS = np.array([1, 3, 4, 1, 2, 1, 3, 1, 2])
COLORS = {
    "High": (0, 0, 255),    # Red
    "Med": (0, 165, 255),   # Orange
    "Low": (255, 255, 0),   # Cyan
    "Seg": (0, 255, 0)      # Green
}

try:
    from models.convnext_model import get_convnext_model
    from utils.transforms import get_basic_transform
except ImportError: pass

# ================= BACKEND LOGIC =================

def load_model():
    print(f"--- Loading Model ({DEVICE}) ---")
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = hf_hub_download(repo_id=HF_ID, filename=HF_FILE, cache_dir=CACHE_DIR)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    
    model = get_convnext_model()
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.to(DEVICE).eval()
    thresh = torch.tensor(ckpt.get("thresholds", [0.5]*9)).to(DEVICE)
    return model, thresh

def get_files(img_id):
    f_img = None
    for ext in ['.jpg', '.png', '.jpeg']:
        p = os.path.join(IMG_DIR, f"{img_id}{ext}")
        if os.path.exists(p): 
            f_img = p
            break
            
    dirs = [CNT_DIR, os.path.join(DATASET_DIR, "contours"), os.path.join(DATASET_DIR, "json")]
    names = [f"{img_id}_contours.json", f"{img_id}.json"]
    f_json = None
    for d in dirs:
        if not os.path.exists(d): continue
        for n in names:
            if os.path.exists(os.path.join(d, n)):
                f_json = os.path.join(d, n); break
        if f_json: break
    return f_img, f_json

def preprocess_cell(img, contour):
    h, w = img.shape[:2]
    x, y, w_b, h_b = cv2.boundingRect(contour)
    pad = 1
    x1, y1 = max(0, x-pad), max(0, y-pad)
    x2, y2 = min(w, x+w_b+pad), min(h, y+h_b+pad)
    
    roi = img[y1:y2, x1:x2].copy()
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour - [x1, y1]], -1, 255, -1)
    rgba = cv2.merge((*cv2.split(roi), mask))
    
    h_c, w_c = rgba.shape[:2]
    scale = (IMG_SIZE * 0.45) / max(h_c, w_c)
    nw, nh = int(w_c * scale), int(h_c * scale)
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    resized = cv2.resize(rgba, (nw, nh), interpolation=interp)
    
    canvas = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255
    xc, yc = (IMG_SIZE - nw)//2, (IMG_SIZE - nh)//2
    rgb, alpha = resized[..., :3], resized[..., 3] / 255.0
    
    dest = canvas[yc:yc+nh, xc:xc+nw]
    for c in range(3): dest[..., c] = (alpha * rgb[..., c] + (1.0-alpha) * dest[..., c])
    canvas[yc:yc+nh, xc:xc+nw] = dest
    return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

def analyze_image(img_id, progress_callback=None):
    f_img, f_json = get_files(img_id)
    if not f_img: return None, "Image not found.", None, None
    if not f_json: return None, "Contour JSON not found.", None, None

    try:
        model, thresholds = load_model()
        tf = get_basic_transform(IMG_SIZE)
        
        img_clean = cv2.imread(f_img)
        img_proc = img_clean.copy()
        
        with open(f_json) as f:
            jdata = json.load(f)
            contours = list(jdata.values()) if isinstance(jdata, dict) else jdata
            
        vis_data = []
        export_data = {}
        
        total = len(contours)
        for i, item in enumerate(contours):
            if progress_callback: progress_callback(i, total)
            
            pts = item.get('points') or item.get('contour')
            cell_id = item.get('id', i)
            if not pts: continue
            cnt = np.array(pts, dtype=np.int32)
            
            pil_input = preprocess_cell(img_proc, cnt)
            with torch.no_grad():
                t = tf(pil_input).unsqueeze(0).to(DEVICE)
                preds = (torch.sigmoid(model(t)) > thresholds).int().cpu().numpy()[0]
            
            score = np.dot(preds, WEIGHTS)
            
            if preds[2] == 1: lvl = "High"
            elif score >= 7: lvl = "High"
            elif score >= 5: lvl = "Med"
            else: lvl = "Low"
            
            vis_data.append({'contour': cnt, 'risk': lvl, 'score': float(score)})
            export_data[f"cell_{cell_id}"] = {"contour": pts, "features": preds.tolist()}
            
        vis_data.sort(key=lambda x: x['score'], reverse=True)
        return img_clean, "Success", vis_data, export_data
        
    except Exception as e:
        return None, str(e), None, None

# ================= GUI CLASSES =================

class ResultWindow:
    def __init__(self, parent, img_clean, analysis_data, img_name):
        self.top = tk.Toplevel(parent)
        self.top.title(f"Analysis Result: {img_name}")
        self.top.geometry("1200x800")
        
        self.img_clean = img_clean
        self.analysis_data = analysis_data
        self.current_vis_cv2 = None 
        
        # Biến điều khiển:
        # - show_segmentation (Hiển thị tất cả contour nền): Mặc định BẬT
        # - show_key_cells (Hiển thị các tế bào quan trọng): Mặc định BẬT
        self.show_segmentation = tk.BooleanVar(value=True)
        self.show_key_cells = tk.BooleanVar(value=True)
        
        # Layout
        main_pane = tk.PanedWindow(self.top, orient=tk.HORIZONTAL, sashwidth=4)
        main_pane.pack(fill=tk.BOTH, expand=True)
        
        self.frame_img = tk.Frame(main_pane, bg="#202020")
        main_pane.add(self.frame_img, stretch="always")
        
        self.lbl_img = tk.Label(self.frame_img, bg="#202020")
        self.lbl_img.pack(fill=tk.BOTH, expand=True)
        
        self.frame_ctrl = tk.Frame(main_pane, width=300, padx=20, pady=20)
        main_pane.add(self.frame_ctrl, stretch="never")
        
        tk.Label(self.frame_ctrl, text="VISUALIZATION CONTROLS", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 15))
        
        # Toggles Group
        group_layers = tk.LabelFrame(self.frame_ctrl, text="Display Layers", padx=15, pady=10)
        group_layers.pack(fill=tk.X, pady=5)
        
        # 1. Segmentation Mask (Hiển thị trước/ưu tiên hiển thị)
        tk.Checkbutton(group_layers, text="Segmentation Mask", 
                       var=self.show_segmentation, font=("Segoe UI", 10), fg="#388e3c",
                       command=self.refresh_image).pack(anchor="w", pady=2)

        # 2. Key Diagnostic Cells (Hiển thị sau/đè lên)
        tk.Checkbutton(group_layers, text="Key Diagnostic Cells", 
                       var=self.show_key_cells, font=("Segoe UI", 10, "bold"), fg="#d32f2f",
                       command=self.refresh_image).pack(anchor="w", pady=2)
        
        # Save Button
        btn_save = tk.Button(self.frame_ctrl, text="Export Image", font=("Segoe UI", 11), 
                             bg="#0078d4", fg="white", relief=tk.FLAT, padx=10, pady=5,
                             command=self.save_current_view)
        btn_save.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        self.top.after(100, self.refresh_image)

    def refresh_image(self):
        if self.img_clean is None: return

        # Luôn bắt đầu từ ảnh gốc sạch
        vis_img = self.img_clean.copy()
        
        # Layer 1: Segmentation Mask (Vẽ trước, nằm dưới)
        if self.show_segmentation.get():
            all_contours = [x['contour'] for x in self.analysis_data]
            cv2.drawContours(vis_img, all_contours, -1, COLORS["Seg"], 1)

        # Layer 2: Key Diagnostic Cells (Vẽ sau, đè lên trên)
        if self.show_key_cells.get():
            # Lấy Top 10 (đã sort)
            key_list = self.analysis_data[:10]
            for item in key_list:
                risk = item['risk']
                cnt = item['contour']
                # Vẽ đậm hơn (thickness=2) để nổi bật trên nền segmentation
                cv2.drawContours(vis_img, [cnt], -1, COLORS[risk], 2)
        
        self.current_vis_cv2 = vis_img
        self.display_cv2_image(vis_img)

    def display_cv2_image(self, cv_img):
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        w_win = self.frame_img.winfo_width()
        h_win = self.frame_img.winfo_height()
        
        if w_win <= 1: w_win = 800
        if h_win <= 1: h_win = 600
        
        w_img, h_img = pil_img.size
        if w_img > 0 and h_img > 0:
            ratio = min(w_win/w_img, h_win/h_img)
            ratio = min(ratio, 1.5) 
            
            new_w = int(w_img * ratio)
            new_h = int(h_img * ratio)
            
            if new_w > 0 and new_h > 0:
                pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
            
        photo = ImageTk.PhotoImage(pil_img)
        self.lbl_img.config(image=photo)
        self.lbl_img.image = photo

    def save_current_view(self):
        path = os.path.join(OUT_DIR, "Exported_View.jpg")
        os.makedirs(OUT_DIR, exist_ok=True)
        if self.current_vis_cv2 is not None:
            cv2.imwrite(path, self.current_vis_cv2)
            messagebox.showinfo("Export Success", f"Image saved to:\n{path}")

class CancerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Thyroid Cancer Diagnosis Assistant")
        self.root.geometry("650x550")
        
        style = ttk.Style()
        style.theme_use('clam')
        
        tk.Label(root, text="THYROID CANCER DIAGNOSIS SYSTEM", font=("Segoe UI", 18, "bold"), pady=20, fg="#2c3e50").pack()
        
        frame_content = tk.Frame(root, padx=30)
        frame_content.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(frame_content, text="Select Biopsy Image:", font=("Segoe UI", 11)).pack(anchor="w")
        
        frame_list = tk.Frame(frame_content, bd=1, relief=tk.SUNKEN)
        frame_list.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = tk.Scrollbar(frame_list)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.listbox = tk.Listbox(frame_list, font=("Consolas", 11), height=12, 
                                  yscrollcommand=scrollbar.set, bd=0, activestyle="none")
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.listbox.yview)
        
        self.load_image_list()
        
        self.btn_run = tk.Button(root, text="START ANALYSIS", font=("Segoe UI", 12, "bold"), 
                                 bg="#0078d4", fg="white", height=2, relief=tk.FLAT,
                                 command=self.run_analysis)
        self.btn_run.pack(fill=tk.X, padx=60, pady=25)
        
        self.progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress.pack(fill=tk.X, padx=30)
        
        self.lbl_status = tk.Label(root, text="Ready", fg="gray", font=("Segoe UI", 9))
        self.lbl_status.pack(pady=5)

    def load_image_list(self):
        if not os.path.exists(IMG_DIR):
            self.listbox.insert(tk.END, "Error: Image directory not found.")
            return
        files = sorted([os.path.splitext(f)[0] for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        for f in files: self.listbox.insert(tk.END, f)

    def run_analysis(self):
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showwarning("Selection Required", "Please select an image from the list.")
            return
        img_id = self.listbox.get(selection[0])
        self.btn_run.config(state=tk.DISABLED, text="PROCESSING...", bg="#cccccc")
        self.progress['value'] = 0
        threading.Thread(target=self.process_thread, args=(img_id,)).start()

    def process_thread(self, img_id):
        self.update_status(f"Loading model & analyzing {img_id}...")
        def progress_cb(curr, total):
            pct = (curr / total) * 100
            self.root.after(0, lambda: self.progress.configure(value=pct))
            self.root.after(0, lambda: self.update_status(f"Analyzing cells: {curr}/{total}"))
        
        img_clean, msg, vis_data, export_data = analyze_image(img_id, progress_cb)
        self.root.after(0, lambda: self.finish_analysis(img_clean, msg, vis_data, export_data, img_id))

    def update_status(self, text): self.lbl_status.config(text=text)

    def finish_analysis(self, img_clean, msg, vis_data, export_data, img_id):
        self.btn_run.config(state=tk.NORMAL, text="START ANALYSIS", bg="#0078d4")
        self.progress['value'] = 100
        
        if img_clean is not None:
            self.update_status("Completed successfully.")
            os.makedirs(JSON_OUT_DIR, exist_ok=True)
            json_path = os.path.join(JSON_OUT_DIR, f"{img_id}_detailed_results.json")
            with open(json_path, 'w') as f: json.dump({img_id: export_data}, f, indent=4)
            print(f"JSON exported: {json_path}")
            ResultWindow(self.root, img_clean, vis_data, img_id)
        else:
            self.update_status("Failed.")
            messagebox.showerror("Processing Error", msg)

if __name__ == "__main__":
    try:
        import models 
        root = tk.Tk()
        app = CancerApp(root)
        root.mainloop()
    except ImportError:
        print("Error: Please run from the repository root directory.")
        input("Enter to exit...")