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

# ================= CẤU HÌNH (CONFIG) =================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path: sys.path.append(ROOT_DIR)

# Auto-detect Dataset
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

if not DATASET_DIR:
    # Fallback cho GUI: Nếu không tìm thấy, dùng đường dẫn tương đối giả định
    DATASET_DIR = os.path.join(ROOT_DIR, "Dataset")

IMG_DIR = os.path.join(DATASET_DIR, "images")
CNT_DIR = os.path.join(DATASET_DIR, "contour")
OUT_DIR = os.path.join(DATASET_DIR, "output", "final_visualization")

# Config Model
HF_ID = "SoftmaxSamurai/ConvNext_SCTC"
HF_FILE = "best_model.pth"
CACHE_DIR = os.path.join(ROOT_DIR, "weights")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

# Config Algo
WEIGHTS = np.array([1, 3, 4, 1, 2, 1, 3, 1, 2])
COLORS = {"High": (0, 0, 255), "Med": (0, 165, 255), "Low": (255, 255, 0)}

# Import Modules
try:
    from models.convnext_model import get_convnext_model
    from utils.transforms import get_basic_transform
except ImportError:
    pass # GUI sẽ báo lỗi sau nếu cần

# ================= PROCESSING FUNCTIONS =================

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
    f_img = next((os.path.join(IMG_DIR, f"{img_id}{e}") for e in ['.jpg', '.png', '.jpeg'] 
                  if os.path.exists(os.path.join(IMG_DIR, f"{img_id}{e}"))), None)
    
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

def process_image(img_id, progress_callback=None):
    f_img, f_json = get_files(img_id)
    if not f_img: return None, "Không tìm thấy ảnh gốc."
    if not f_json: return None, "Không tìm thấy file contour JSON."

    try:
        model, thresholds = load_model()
        tf = get_basic_transform(IMG_SIZE)
        os.makedirs(OUT_DIR, exist_ok=True)
        
        img_vis = cv2.imread(f_img)
        img_proc = cv2.imread(f_img)
        
        with open(f_json) as f:
            jdata = json.load(f)
            contours = list(jdata.values()) if isinstance(jdata, dict) else jdata
            
        total = len(contours)
        for i, item in enumerate(contours):
            if progress_callback: progress_callback(i, total)
            
            pts = item.get('points') or item.get('contour')
            if not pts: continue
            cnt = np.array(pts, dtype=np.int32)
            
            pil_input = preprocess_cell(img_proc, cnt)
            with torch.no_grad():
                t = tf(pil_input).unsqueeze(0).to(DEVICE)
                preds = (torch.sigmoid(model(t)) > thresholds).int().cpu().numpy()[0]
            
            score = np.dot(preds, WEIGHTS)
            if preds[2] == 1 or score >= 7: lvl = "High"
            elif score >= 5: lvl = "Med"
            else: lvl = "Low"
            
            cv2.drawContours(img_vis, [cnt], -1, COLORS[lvl], 2)
            
        save_path = os.path.join(OUT_DIR, f"Vis_{img_id}_Final.jpg")
        cv2.imwrite(save_path, img_vis)
        return save_path, "Success"
        
    except Exception as e:
        return None, str(e)

# ================= GUI APPLICATION =================

class CancerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cancer Cell Visualization Tool")
        self.root.geometry("600x450")
        
        # Header
        tk.Label(root, text="HỆ THỐNG CHẨN ĐOÁN TẾ BÀO UNG THƯ", font=("Arial", 16, "bold"), pady=10).pack()
        
        # Frame chọn ảnh
        frame_list = tk.Frame(root)
        frame_list.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
        
        tk.Label(frame_list, text="Danh sách ảnh trong Dataset:", font=("Arial", 10)).pack(anchor="w")
        
        # Scrollbar & Listbox
        scrollbar = tk.Scrollbar(frame_list)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.listbox = tk.Listbox(frame_list, font=("Arial", 11), height=10)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.listbox.yview)
        
        # Load danh sách ảnh
        self.load_image_list()
        
        # Nút chạy
        self.btn_run = tk.Button(root, text="CHẠY PHÂN TÍCH", font=("Arial", 12, "bold"), 
                                 bg="#007acc", fg="white", height=2, command=self.run_analysis)
        self.btn_run.pack(fill=tk.X, padx=50, pady=15)
        
        # Progress Bar
        self.progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.lbl_status = tk.Label(root, text="Sẵn sàng", fg="gray")
        self.progress.pack(fill=tk.X, padx=20)
        self.lbl_status.pack(pady=5)

    def load_image_list(self):
        if not os.path.exists(IMG_DIR):
            self.listbox.insert(tk.END, "Lỗi: Không tìm thấy thư mục images!")
            return
        
        files = sorted([os.path.splitext(f)[0] for f in os.listdir(IMG_DIR) 
                       if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        for f in files:
            self.listbox.insert(tk.END, f)

    def run_analysis(self):
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn một ảnh để xử lý!")
            return
            
        img_id = self.listbox.get(selection[0])
        
        # Disable nút bấm
        self.btn_run.config(state=tk.DISABLED, text="Đang xử lý...")
        self.progress['value'] = 0
        
        # Chạy thread riêng để không treo GUI
        threading.Thread(target=self.process_thread, args=(img_id,)).start()

    def process_thread(self, img_id):
        self.update_status(f"Đang tải model và xử lý {img_id}...")
        
        def progress_cb(curr, total):
            pct = (curr / total) * 100
            self.root.after(0, lambda: self.progress.configure(value=pct))
            self.root.after(0, lambda: self.update_status(f"Đang phân tích tế bào: {curr}/{total}"))

        out_path, msg = process_image(img_id, progress_cb)
        
        self.root.after(0, lambda: self.finish_analysis(out_path, msg))

    def update_status(self, text):
        self.lbl_status.config(text=text)

    def finish_analysis(self, out_path, msg):
        self.btn_run.config(state=tk.NORMAL, text="CHẠY PHÂN TÍCH")
        self.progress['value'] = 100
        
        if out_path:
            self.update_status("Hoàn tất!")
            # Hiển thị ảnh kết quả trong cửa sổ mới
            self.show_result_window(out_path)
        else:
            self.update_status("Thất bại")
            messagebox.showerror("Lỗi", f"Có lỗi xảy ra:\n{msg}")

    def show_result_window(self, img_path):
        top = tk.Toplevel(self.root)
        top.title("Kết quả Phân tích")
        top.geometry("800x600")
        
        # Load ảnh và resize cho vừa cửa sổ
        img = Image.open(img_path)
        # Resize giữ tỷ lệ
        display_size = (780, 550)
        img.thumbnail(display_size, Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        
        lbl_img = tk.Label(top, image=photo)
        lbl_img.image = photo # Giữ tham chiếu để không bị garbage collector xóa
        lbl_img.pack(expand=True)
        
        btn_close = tk.Button(top, text="Đóng", command=top.destroy)
        btn_close.pack(pady=5)

if __name__ == "__main__":
    # Kiểm tra module trước khi chạy GUI
    try:
        import models
        root = tk.Tk()
        app = CancerApp(root)
        root.mainloop()
    except ImportError:
        print("Lỗi: Vui lòng chạy script này từ thư mục gốc của repo (nơi chứa folder models).")
        input("Nhấn Enter để thoát...")