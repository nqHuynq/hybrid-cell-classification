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

# Algorithm Config
WEIGHTS = np.array([1, 3, 4, 1, 2, 1, 3, 1, 2])
FEATURE_NAMES = [
    "Elongation", "Grooves", "Inclusion", 
    "Glassy", "Crowding", "Wrinkled", 
    "Enlargement", "Irregular", "Overlapping"
]

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

def analyze_single_image(img_id, model, thresholds, transform, progress_callback=None):
    f_img, f_json = get_files(img_id)
    if not f_img: return None, f"Image {img_id} not found.", None, None
    if not f_json: return None, f"JSON {img_id} not found.", None, None

    try:
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
                t = transform(pil_input).unsqueeze(0).to(DEVICE)
                preds = (torch.sigmoid(model(t)) > thresholds).int().cpu().numpy()[0]
            
            score = np.dot(preds, WEIGHTS)
            if preds[2] == 1 or score >= 7: lvl = "High"
            elif score >= 5: lvl = "Med"
            else: lvl = "Low"
            
            # Convert preds (0/1) to list of feature names
            active_features = [FEATURE_NAMES[idx] for idx, val in enumerate(preds) if val == 1]
            
            cell_info = {
                'id': cell_id,
                'contour': cnt, 
                'risk': lvl, 
                'score': float(score),
                'features_list': active_features
            }
            vis_data.append(cell_info)
            
            export_data[f"cell_{cell_id}"] = {
                "contour": pts, 
                "features_binary": preds.tolist(),
                "features_text": active_features
            }
            
        vis_data.sort(key=lambda x: x['score'], reverse=True)
        return img_clean, "Success", vis_data, export_data
        
    except Exception as e:
        return None, str(e), None, None

# ================= GUI CLASSES =================

class ToolTip(object):
    """Tooltip class để hiện thông tin khi hover."""
    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text, x_root, y_root):
        "Display text in tooltip window"
        if self.tipwindow or not text:
            return
        # Tọa độ hiển thị (lệch một chút so với chuột)
        x, y = x_root + 20, y_root + 20
        
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1) # Bỏ viền cửa sổ
        tw.wm_geometry("+%d+%d" % (x, y))
        
        label = tk.Label(tw, text=text, justify=tk.LEFT,
                       background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                       font=("Segoe UI", 9))
        label.pack(ipadx=5, ipady=2)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

class ResultWindow:
    def __init__(self, parent, results_list):
        self.top = tk.Toplevel(parent)
        self.top.title(f"Analysis Results ({len(results_list)} Images)")
        self.top.geometry("1300x850")
        
        self.results = results_list
        self.current_idx = 0
        self.current_vis_cv2 = None 
        
        self.show_segmentation = tk.BooleanVar(value=True)
        self.show_key_cells = tk.BooleanVar(value=True)
        
        # Tỷ lệ scale hiển thị ảnh (quan trọng cho việc map chuột)
        self.display_scale = 1.0 
        self.display_offset_x = 0
        self.display_offset_y = 0
        
        # Layout
        main_pane = tk.PanedWindow(self.top, orient=tk.HORIZONTAL, sashwidth=4)
        main_pane.pack(fill=tk.BOTH, expand=True)
        
        self.frame_img = tk.Frame(main_pane, bg="#202020")
        main_pane.add(self.frame_img, stretch="always")
        
        self.lbl_img = tk.Label(self.frame_img, bg="#202020")
        self.lbl_img.pack(fill=tk.BOTH, expand=True)
        
        # Gắn sự kiện chuột cho tooltip
        self.tooltip = ToolTip(self.lbl_img)
        self.lbl_img.bind("<Motion>", self.on_mouse_move)
        
        # Control Frame
        self.frame_ctrl = tk.Frame(main_pane, width=300, padx=20, pady=20)
        main_pane.add(self.frame_ctrl, stretch="never")
        
        # --- Controls UI ---
        if len(self.results) > 1:
            frame_nav = tk.Frame(self.frame_ctrl)
            frame_nav.pack(fill=tk.X, pady=(0, 15))
            self.btn_prev = tk.Button(frame_nav, text="<< Prev", command=self.prev_image)
            self.btn_prev.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
            self.lbl_counter = tk.Label(frame_nav, text=f"1 / {len(self.results)}", font=("Segoe UI", 10, "bold"))
            self.lbl_counter.pack(side=tk.LEFT, padx=5)
            self.btn_next = tk.Button(frame_nav, text="Next >>", command=self.next_image)
            self.btn_next.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
            tk.Label(self.frame_ctrl, text="", height=1).pack() 

        self.lbl_title = tk.Label(self.frame_ctrl, text="", font=("Segoe UI", 14, "bold", "underline"))
        self.lbl_title.pack(anchor="w", pady=(0, 15))

        tk.Label(self.frame_ctrl, text="VISUALIZATION CONTROLS", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        
        group_layers = tk.LabelFrame(self.frame_ctrl, text="Display Layers", padx=15, pady=10)
        group_layers.pack(fill=tk.X, pady=5)
        
        tk.Checkbutton(group_layers, text="Segmentation Mask", 
                       var=self.show_segmentation, font=("Segoe UI", 10), fg="#388e3c",
                       command=self.refresh_image).pack(anchor="w", pady=2)
        
        tk.Checkbutton(group_layers, text="Key Diagnostic Cells", 
                       var=self.show_key_cells, font=("Segoe UI", 10, "bold"), fg="#d32f2f",
                       command=self.refresh_image).pack(anchor="w", pady=2)
        
        tk.Label(group_layers, text="(Hover over red cells to see details)", 
                 font=("Segoe UI", 9, "italic"), fg="gray").pack(anchor="w", pady=2)
        
        btn_save = tk.Button(self.frame_ctrl, text="Export Current Image", font=("Segoe UI", 11), 
                             bg="#0078d4", fg="white", relief=tk.FLAT, padx=10, pady=5,
                             command=self.save_current_view)
        btn_save.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        self.load_current_data()

    # --- MOUSE EVENT HANDLER ---
    def on_mouse_move(self, event):
        """Xử lý sự kiện di chuột để hiện tooltip."""
        if not self.show_key_cells.get(): 
            self.tooltip.hidetip()
            return

        # Lấy tọa độ trên widget Label
        mx, my = event.x, event.y
        
        # Tính toán tọa độ tương ứng trên ảnh gốc (dựa trên cách ảnh được resize/center)
        # img_w_display = original_w * scale
        # offset_x = (label_w - img_w_display) / 2
        
        if not hasattr(self, 'img_clean'): return
        orig_h, orig_w = self.img_clean.shape[:2]
        
        # Tính tọa độ gốc
        # x_orig = (mx - offset_x) / scale
        
        # Cần tính lại các tham số hiển thị hiện tại từ widget
        w_win = self.lbl_img.winfo_width()
        h_win = self.lbl_img.winfo_height()
        
        if w_win <= 1 or h_win <= 1: return
        
        # Logic resize giống hệt hàm display_cv2_image
        ratio = min(w_win/orig_w, h_win/orig_h)
        ratio = min(ratio, 1.5)
        
        new_w = int(orig_w * ratio)
        new_h = int(orig_h * ratio)
        
        offset_x = (w_win - new_w) // 2
        offset_y = (h_win - new_h) // 2
        
        # Kiểm tra chuột có nằm trong vùng ảnh không
        if mx < offset_x or mx > offset_x + new_w or my < offset_y or my > offset_y + new_h:
            self.tooltip.hidetip()
            return
            
        # Map về tọa độ gốc
        real_x = int((mx - offset_x) / ratio)
        real_y = int((my - offset_y) / ratio)
        
        # Kiểm tra va chạm (Hit Test) với các contour Key Cells
        found_cell = None
        # Chỉ check Top 10 (Key Cells)
        key_list = self.analysis_data['data'][:10]
        
        for item in key_list:
            # pointPolygonTest trả về >= 0 nếu điểm nằm trong hoặc trên biên contour
            dist = cv2.pointPolygonTest(item['contour'], (real_x, real_y), False)
            if dist >= 0:
                found_cell = item
                break
        
        if found_cell:
            # Tạo nội dung tooltip
            feats = "\n- ".join(found_cell['features_list'])
            if not feats: feats = "None"
            tip_text = (f"Cell ID: {item.get('id', '?')}\n"
                        f"Risk: {found_cell['risk']}\n"
                        f"Score: {found_cell['score']:.1f}\n"
                        f"Features:\n- {feats}")
            self.tooltip.showtip(tip_text, event.x_root, event.y_root)
        else:
            self.tooltip.hidetip()

    # --- NAVIGATION ---
    def prev_image(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_current_data()

    def next_image(self):
        if self.current_idx < len(self.results) - 1:
            self.current_idx += 1
            self.load_current_data()

    def load_current_data(self):
        data = self.results[self.current_idx]
        self.img_clean = data['img']
        self.analysis_data = data # Lưu cả dict chứa 'data' (list cells)
        
        self.lbl_title.config(text=f"Image: {data['id']}")
        if len(self.results) > 1:
            self.lbl_counter.config(text=f"{self.current_idx + 1} / {len(self.results)}")
            self.btn_prev.config(state=tk.NORMAL if self.current_idx > 0 else tk.DISABLED)
            self.btn_next.config(state=tk.NORMAL if self.current_idx < len(self.results) - 1 else tk.DISABLED)
        self.refresh_image()

    def refresh_image(self):
        if self.img_clean is None: return
        vis_img = self.img_clean.copy()
        cell_list = self.analysis_data['data'] # List các cell đã phân tích
        
        if self.show_segmentation.get():
            all_contours = [x['contour'] for x in cell_list]
            cv2.drawContours(vis_img, all_contours, -1, COLORS["Seg"], 1)

        if self.show_key_cells.get():
            key_list = cell_list[:10]
            for item in key_list:
                cv2.drawContours(vis_img, [item['contour']], -1, COLORS[item['risk']], 2)
        
        self.current_vis_cv2 = vis_img
        self.display_cv2_image(vis_img)

    def display_cv2_image(self, cv_img):
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        w_win = self.frame_img.winfo_width() or 900
        h_win = self.frame_img.winfo_height() or 700
        if w_win <= 1: w_win = 800
        if h_win <= 1: h_win = 600
        
        w_img, h_img = pil_img.size
        if w_img > 0 and h_img > 0:
            ratio = min(w_win/w_img, h_win/h_img)
            ratio = min(ratio, 1.5) 
            new_size = (int(w_img*ratio), int(h_img*ratio))
            if new_size[0] > 0: pil_img = pil_img.resize(new_size, Image.LANCZOS)
            
        photo = ImageTk.PhotoImage(pil_img)
        self.lbl_img.config(image=photo)
        self.lbl_img.image = photo

    def save_current_view(self):
        current_id = self.results[self.current_idx]['id']
        path = os.path.join(OUT_DIR, f"Export_{current_id}.jpg")
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
        style.configure("green.Horizontal.TProgressbar", background='#4CAF50', troughcolor='#E0E0E0')
        
        tk.Label(root, text="THYROID CANCER DIAGNOSIS SYSTEM", font=("Segoe UI", 18, "bold"), pady=20, fg="#2c3e50").pack()
        
        frame_content = tk.Frame(root, padx=30)
        frame_content.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(frame_content, text="Select Biopsy Images (Multi-select):", font=("Segoe UI", 11)).pack(anchor="w")
        
        frame_list = tk.Frame(frame_content, bd=1, relief=tk.SUNKEN)
        frame_list.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = tk.Scrollbar(frame_list)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.listbox = tk.Listbox(frame_list, font=("Consolas", 11), height=12, selectmode=tk.EXTENDED,
                                  yscrollcommand=scrollbar.set, bd=0, activestyle="none")
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.listbox.yview)
        
        self.load_image_list()
        
        self.btn_run = tk.Button(root, text="START ANALYSIS", font=("Segoe UI", 12, "bold"), 
                                 bg="#0078d4", fg="white", height=2, relief=tk.FLAT,
                                 command=self.run_analysis)
        self.btn_run.pack(fill=tk.X, padx=60, pady=25)
        
        self.progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=100, mode='determinate', style="green.Horizontal.TProgressbar")
        self.progress.pack(fill=tk.X, padx=30)
        
        self.lbl_status = tk.Label(root, text="Ready (Hold Ctrl/Shift to select multiple images)", fg="gray", font=("Segoe UI", 9))
        self.lbl_status.pack(pady=5)

    def load_image_list(self):
        if not os.path.exists(IMG_DIR):
            self.listbox.insert(tk.END, "Error: Image directory not found.")
            return
        files = sorted([os.path.splitext(f)[0] for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        for f in files: self.listbox.insert(tk.END, f)

    def run_analysis(self):
        indices = self.listbox.curselection()
        if not indices:
            messagebox.showwarning("Selection Required", "Please select at least one image.")
            return
        img_ids = [self.listbox.get(i) for i in indices]
        self.btn_run.config(state=tk.DISABLED, text="PROCESSING...", bg="#cccccc")
        self.progress['value'] = 0
        threading.Thread(target=self.process_thread, args=(img_ids,)).start()

    def process_thread(self, img_ids):
        self.update_status(f"Loading model...")
        final_results = [] 
        errors = []
        try:
            model, thresholds = load_model()
            tf = get_basic_transform(IMG_SIZE)
            os.makedirs(JSON_OUT_DIR, exist_ok=True)
            total_imgs = len(img_ids)
            for idx, img_id in enumerate(img_ids):
                self.update_status(f"Processing {img_id} ({idx+1}/{total_imgs})...")
                def img_progress_cb(curr, total_cells):
                    base_pct = (idx / total_imgs) * 100
                    current_img_pct = (curr / total_cells) * (100 / total_imgs)
                    total_pct = base_pct + current_img_pct
                    self.root.after(0, lambda: self.progress.configure(value=total_pct))

                img_clean, msg, vis_data, export_data = analyze_single_image(img_id, model, thresholds, tf, img_progress_cb)
                
                if img_clean is not None:
                    json_path = os.path.join(JSON_OUT_DIR, f"{img_id}_detailed_results.json")
                    with open(json_path, 'w') as f: json.dump({img_id: export_data}, f, indent=4)
                    final_results.append({'id': img_id, 'img': img_clean, 'data': vis_data})
                else:
                    errors.append(f"{img_id}: {msg}")

            self.root.after(0, lambda: self.finish_analysis(final_results, errors))
        except Exception as e:
            self.root.after(0, lambda: self.finish_analysis([], [str(e)]))

    def update_status(self, text): self.lbl_status.config(text=text)

    def finish_analysis(self, final_results, errors):
        self.btn_run.config(state=tk.NORMAL, text="START ANALYSIS", bg="#0078d4")
        self.progress['value'] = 100
        if final_results:
            self.update_status(f"Completed {len(final_results)} images.")
            ResultWindow(self.root, final_results) 
        else:
            self.update_status("Failed.")
        if errors:
            err_msg = "\n".join(errors[:5]) + ("\n..." if len(errors)>5 else "")
            messagebox.showwarning("Process Warnings", f"Some images failed:\n{err_msg}")

if __name__ == "__main__":
    try:
        import models 
        root = tk.Tk()
        app = CancerApp(root)
        root.mainloop()
    except ImportError:
        print("Error: Please run from the repository root directory.")
        input("Enter to exit...")