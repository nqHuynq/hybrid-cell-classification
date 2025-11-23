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

# Tự động tìm Dataset
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
    DATASET_DIR = os.path.join(ROOT_DIR, "Dataset")

IMG_DIR = os.path.join(DATASET_DIR, "images")
CNT_DIR = os.path.join(DATASET_DIR, "contour")
OUT_DIR = os.path.join(DATASET_DIR, "output", "final_visualization")

HF_ID = "SoftmaxSamurai/ConvNext_SCTC"
HF_FILE = "best_model.pth"
CACHE_DIR = os.path.join(ROOT_DIR, "weights")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

WEIGHTS = np.array([1, 3, 4, 1, 2, 1, 3, 1, 2])
# BGR Colors
COLORS = {"High": (0, 0, 255), "Med": (0, 165, 255), "Low": (255, 255, 0)} 

try:
    from models.convnext_model import get_convnext_model
    from utils.transforms import get_basic_transform
except ImportError: pass

# ================= UTILS =================
def bgr_to_hex(bgr):
    """Chuyển màu BGR (OpenCV) sang Hex (Tkinter)."""
    b, g, r = bgr
    return f'#{r:02x}{g:02x}{b:02x}'

# ================= LOGIC XỬ LÝ =================

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
    if not f_img: return None, "Không tìm thấy ảnh gốc.", None
    if not f_json: return None, "Không tìm thấy file contour JSON.", None

    try:
        model, thresholds = load_model()
        tf = get_basic_transform(IMG_SIZE)
        
        img_clean = cv2.imread(f_img)
        img_proc = img_clean.copy()
        
        with open(f_json) as f:
            jdata = json.load(f)
            contours = list(jdata.values()) if isinstance(jdata, dict) else jdata
            
        analysis_results = []
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
            
            analysis_results.append({'contour': cnt, 'risk': lvl})
            
        return img_clean, "Success", analysis_results
        
    except Exception as e:
        return None, str(e), None

# ================= GUI CLASSES =================

class StatRow(tk.Frame):
    """Widget hiển thị một dòng thống kê với chấm màu."""
    def __init__(self, parent, color_hex, label_text, initial_value=0):
        super().__init__(parent)
        self.color = color_hex
        
        # Chấm tròn màu
        self.canvas = tk.Canvas(self, width=20, height=20, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, padx=(0, 5))
        # Vẽ hình tròn có viền đen
        self.canvas.create_oval(2, 2, 18, 18, fill=color_hex, outline="black")
        
        # Label text
        self.lbl_text = tk.Label(self, text=label_text, font=("Arial", 11))
        self.lbl_text.pack(side=tk.LEFT)
        
        # Label giá trị
        self.lbl_val = tk.Label(self, text=str(initial_value), font=("Arial", 11, "bold"))
        self.lbl_val.pack(side=tk.RIGHT, padx=(5, 0))

    def update_value(self, val):
        self.lbl_val.config(text=str(val))

class ResultWindow:
    def __init__(self, parent, img_clean, analysis_data, img_name):
        self.top = tk.Toplevel(parent)
        self.top.title(f"Kết quả: {img_name}")
        self.top.geometry("1200x800")
        
        self.img_clean = img_clean
        self.analysis_data = analysis_data
        self.current_vis_cv2 = None 
        
        self.show_high = tk.BooleanVar(value=True)
        self.show_med = tk.BooleanVar(value=True)
        self.show_low = tk.BooleanVar(value=True)
        
        # --- Layout ---
        main_pane = tk.PanedWindow(self.top, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)
        
        self.frame_img = tk.Frame(main_pane, bg="black")
        main_pane.add(self.frame_img, stretch="always")
        
        self.lbl_img = tk.Label(self.frame_img, bg="black")
        self.lbl_img.pack(fill=tk.BOTH, expand=True)
        
        self.frame_ctrl = tk.Frame(main_pane, width=320, padx=20, pady=20)
        main_pane.add(self.frame_ctrl, stretch="never")
        
        # --- Điều khiển ---
        tk.Label(self.frame_ctrl, text="BỘ LỌC HIỂN THỊ", font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 10))
        
        group_filter = tk.LabelFrame(self.frame_ctrl, text="Tùy chọn", padx=10, pady=10)
        group_filter.pack(fill=tk.X, pady=5)
        
        tk.Checkbutton(group_filter, text="Nguy hiểm cao (Đỏ)", var=self.show_high, 
                       font=("Arial", 10), fg="red", command=self.refresh_image).pack(anchor="w")
        tk.Checkbutton(group_filter, text="Cảnh báo (Cam)", var=self.show_med, 
                       font=("Arial", 10), fg="#FF8C00", command=self.refresh_image).pack(anchor="w")
        tk.Checkbutton(group_filter, text="An toàn (Xanh)", var=self.show_low, 
                       font=("Arial", 10), fg="blue", command=self.refresh_image).pack(anchor="w")
        
        ttk.Separator(self.frame_ctrl, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=20)
        
        # --- Thống kê Số lượng (Giao diện mới) ---
        tk.Label(self.frame_ctrl, text="THỐNG KÊ SỐ LƯỢNG:", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 10))
        
        self.frame_stats = tk.Frame(self.frame_ctrl)
        self.frame_stats.pack(fill=tk.X)
        
        # Tạo các dòng thống kê với màu sắc chuẩn
        self.row_high = StatRow(self.frame_stats, bgr_to_hex(COLORS["High"]), "Đỏ (Cao):")
        self.row_high.pack(fill=tk.X, pady=2)
        
        self.row_med = StatRow(self.frame_stats, bgr_to_hex(COLORS["Med"]), "Cam (Vừa):")
        self.row_med.pack(fill=tk.X, pady=2)
        
        self.row_low = StatRow(self.frame_stats, bgr_to_hex(COLORS["Low"]), "Xanh (Thấp):")
        self.row_low.pack(fill=tk.X, pady=2)
        
        # Đường gạch ngang tổng kết
        tk.Label(self.frame_stats, text="-----------------------------", fg="gray").pack(fill=tk.X)
        
        # Dòng tổng cộng
        self.frame_total = tk.Frame(self.frame_stats)
        self.frame_total.pack(fill=tk.X, pady=5)
        tk.Label(self.frame_total, text="Tổng cộng:", font=("Arial", 12)).pack(side=tk.LEFT)
        self.lbl_total_val = tk.Label(self.frame_total, text="0", font=("Arial", 12, "bold"))
        self.lbl_total_val.pack(side=tk.RIGHT)
        
        # Save Button
        btn_save = tk.Button(self.frame_ctrl, text="Lưu Ảnh Kết Quả", font=("Arial", 11, "bold"), 
                             bg="#4CAF50", fg="white", height=2, command=self.save_current_view)
        btn_save.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        self.refresh_image()

    def refresh_image(self):
        vis_img = self.img_clean.copy()
        counts = {'High': 0, 'Med': 0, 'Low': 0}
        
        for item in self.analysis_data:
            risk = item['risk']
            counts[risk] += 1
            
            should_draw = False
            if risk == 'High' and self.show_high.get(): should_draw = True
            elif risk == 'Med' and self.show_med.get(): should_draw = True
            elif risk == 'Low' and self.show_low.get(): should_draw = True
            
            if should_draw:
                cv2.drawContours(vis_img, [item['contour']], -1, COLORS[risk], 2)
        
        # Cập nhật số liệu thống kê lên UI mới
        self.row_high.update_value(counts['High'])
        self.row_med.update_value(counts['Med'])
        self.row_low.update_value(counts['Low'])
        self.lbl_total_val.config(text=str(len(self.analysis_data)))
        
        self.current_vis_cv2 = vis_img
        self.display_cv2_image(vis_img)

    def display_cv2_image(self, cv_img):
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        w_win = self.frame_img.winfo_width() or 800
        h_win = self.frame_img.winfo_height() or 600
        
        w_img, h_img = pil_img.size
        if w_img > 0 and h_img > 0:
            ratio = min(w_win/w_img, h_win/h_img)
            new_size = (int(w_img*ratio), int(h_img*ratio))
            if new_size[0] > 0 and new_size[1] > 0:
                pil_img = pil_img.resize(new_size, Image.LANCZOS)
            
        photo = ImageTk.PhotoImage(pil_img)
        self.lbl_img.config(image=photo)
        self.lbl_img.image = photo

    def save_current_view(self):
        path = os.path.join(OUT_DIR, "Saved_View.jpg")
        os.makedirs(OUT_DIR, exist_ok=True)
        if self.current_vis_cv2 is not None:
            cv2.imwrite(path, self.current_vis_cv2)
            messagebox.showinfo("Đã lưu", f"Đã lưu ảnh tại:\n{path}")

class CancerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cancer Cell Visualization Tool")
        self.root.geometry("600x500")
        
        tk.Label(root, text="HỆ THỐNG CHẨN ĐOÁN TẾ BÀO UNG THƯ", font=("Arial", 18, "bold"), pady=15).pack()
        
        frame_content = tk.Frame(root, padx=20)
        frame_content.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(frame_content, text="Chọn ảnh bệnh phẩm:", font=("Arial", 12)).pack(anchor="w")
        
        frame_list = tk.Frame(frame_content)
        frame_list.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = tk.Scrollbar(frame_list)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.listbox = tk.Listbox(frame_list, font=("Arial", 11), height=10, yscrollcommand=scrollbar.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.listbox.yview)
        
        self.load_image_list()
        
        self.btn_run = tk.Button(root, text="BẮT ĐẦU PHÂN TÍCH", font=("Arial", 12, "bold"), 
                                 bg="#007acc", fg="white", height=2, command=self.run_analysis)
        self.btn_run.pack(fill=tk.X, padx=50, pady=20)
        
        self.progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress.pack(fill=tk.X, padx=20)
        
        self.lbl_status = tk.Label(root, text="Sẵn sàng", fg="gray")
        self.lbl_status.pack(pady=5)

    def load_image_list(self):
        if not os.path.exists(IMG_DIR):
            self.listbox.insert(tk.END, "Lỗi: Không tìm thấy thư mục images!")
            return
        files = sorted([os.path.splitext(f)[0] for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        for f in files: self.listbox.insert(tk.END, f)

    def run_analysis(self):
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showwarning("Chưa chọn ảnh", "Vui lòng chọn một ảnh!")
            return
        img_id = self.listbox.get(selection[0])
        self.btn_run.config(state=tk.DISABLED, text="Đang xử lý...", bg="#cccccc")
        self.progress['value'] = 0
        threading.Thread(target=self.process_thread, args=(img_id,)).start()

    def process_thread(self, img_id):
        self.update_status(f"Đang tải model và xử lý {img_id}...")
        def progress_cb(curr, total):
            pct = (curr / total) * 100
            self.root.after(0, lambda: self.progress.configure(value=pct))
            self.root.after(0, lambda: self.update_status(f"Đang phân tích: {curr}/{total}"))
        img_clean, msg, results = analyze_image(img_id, progress_cb)
        self.root.after(0, lambda: self.finish_analysis(img_clean, msg, results, img_id))

    def update_status(self, text): self.lbl_status.config(text=text)

    def finish_analysis(self, img_clean, msg, results, img_id):
        self.btn_run.config(state=tk.NORMAL, text="BẮT ĐẦU PHÂN TÍCH", bg="#007acc")
        self.progress['value'] = 100
        if img_clean is not None:
            self.update_status("Hoàn tất!")
            ResultWindow(self.root, img_clean, results, img_id)
        else:
            self.update_status("Lỗi!")
            messagebox.showerror("Lỗi", msg)

if __name__ == "__main__":
    try:
        import models 
        root = tk.Tk()
        app = CancerApp(root)
        root.mainloop()
    except ImportError:
        print("Lỗi: Hãy chạy từ thư mục gốc của repo.")
        input("Enter...")