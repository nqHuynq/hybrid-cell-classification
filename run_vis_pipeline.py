import sys
import os
import torch
import json
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from huggingface_hub import hf_hub_download

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG (SYSTEM CONFIG)
# ==========================================

# Đường dẫn Repo gốc (để load model)
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../hybrid-cell-classification"))
if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)

# Import Module từ Repo gốc
try:
    from models.convnext_model import get_convnext_model
    from utils.transforms import get_basic_transform
except ImportError as e:
    print("Lỗi: Không load được module từ hybrid-cell-classification.")
    raise e

# Đường dẫn dữ liệu
INPUT_CROPS_DIR = r"E:\Work\Github\vnu-tool\Dataset\step2_model_input_224"
IMAGE_DIR_ROOT = r"E:\Work\Github\vnu-tool\Dataset\images"
CONTOUR_DIR_ROOT = r"E:\Work\Github\vnu-tool\Dataset\contour"
OUTPUT_DIR = r"E:\Work\Github\vnu-tool\Dataset\visualization_output"

# Cấu hình Model HF
HF_REPO_ID = "SoftmaxSamurai/ConvNext_SCTC"
HF_FILENAME = "best_model.pth"
CACHE_DIR = "../weights"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

# --- CẤU HÌNH CHẠY ---
TARGET_IMAGE_ID = "B5_150" # Để None nếu muốn chạy tất cả

# ==========================================
# 2. CẤU HÌNH THUẬT TOÁN (ALGORITHM CONFIG)
# ==========================================

# Trọng số [Elongation, Grooves, Inclusion, Glassy, Crowding, Wrinkled, Enlargement, Irregular, Overlapping]
FEATURE_WEIGHTS = np.array([1, 3, 4, 1, 2, 1, 3, 1, 2])

# Ngưỡng phân loại
MIN_DISPLAY_SCORE = 5 
HIGH_RISK_SCORE = 7

# Bảng màu (BGR) - Đậm nét
COLOR_HIGH_RISK = (0, 0, 255)     # Đỏ
COLOR_MEDIUM_RISK = (0, 165, 255) # Cam
COLOR_LOW_RISK = (255, 255, 0)    # Xanh
THICKNESS = 2

# ==========================================
# 3. CÁC HÀM XỬ LÝ (FUNCTIONS)
# ==========================================

def load_ai_model():
    """Load model và thresholds từ file weights."""
    print(f"--- Đang khởi tạo AI Model trên {DEVICE} ---")
    os.makedirs(CACHE_DIR, exist_ok=True)
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME, cache_dir=CACHE_DIR)
    
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    # Lấy thresholds
    if "thresholds" in checkpoint:
        thresholds = torch.tensor(checkpoint["thresholds"]).to(DEVICE)
    else:
        thresholds = torch.full((9,), 0.5).to(DEVICE)

    # Khởi tạo model
    model = get_convnext_model() 
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    
    return model, thresholds

def get_risk_color(pred_vector, score):
    """Quyết định màu sắc dựa trên vector và điểm."""
    # Ưu tiên Inclusion (index 2) -> Đỏ
    if pred_vector[2] == 1:
        return COLOR_HIGH_RISK, "High"
    
    if score >= HIGH_RISK_SCORE:
        return COLOR_HIGH_RISK, "High"
    elif score >= MIN_DISPLAY_SCORE:
        return COLOR_MEDIUM_RISK, "Medium"
    else:
        return COLOR_LOW_RISK, "Low"

def find_files(img_name):
    """Tìm đường dẫn ảnh gốc và file contour."""
    # 1. Tìm ảnh gốc
    img_path = None
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        temp = os.path.join(IMAGE_DIR_ROOT, f"{img_name}{ext}")
        if os.path.exists(temp):
            img_path = temp
            break
            
    # 2. Tìm contour json
    contour_path = None
    possible_dirs = [CONTOUR_DIR_ROOT, os.path.join(CONTOUR_DIR_ROOT, "contours")]
    possible_names = [f"{img_name}_contours.json", f"{img_name}.json"]
    
    for d in possible_dirs:
        if not os.path.exists(d): continue
        for fname in possible_names:
            temp = os.path.join(d, fname)
            if os.path.exists(temp):
                contour_path = temp
                break
        if contour_path: break
        
    return img_path, contour_path

# ==========================================
# 4. QUY TRÌNH CHÍNH (MAIN PIPELINE)
# ==========================================

def run_pipeline():
    # A. Setup
    if not os.path.exists(INPUT_CROPS_DIR):
        print(f"Lỗi: Không tìm thấy thư mục crop tại {INPUT_CROPS_DIR}")
        return

    model, thresholds = load_ai_model()
    transform = get_basic_transform(IMG_SIZE)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Lọc danh sách ảnh cần chạy
    if TARGET_IMAGE_ID:
        print(f"--> CHẾ ĐỘ SINGLE MODE: {TARGET_IMAGE_ID}")
        target_path = os.path.join(INPUT_CROPS_DIR, TARGET_IMAGE_ID)
        if not os.path.exists(target_path):
            print(f"Không tìm thấy dữ liệu crop cho {TARGET_IMAGE_ID}")
            return
        img_folders = [TARGET_IMAGE_ID]
    else:
        img_folders = [d for d in os.listdir(INPUT_CROPS_DIR) if os.path.isdir(os.path.join(INPUT_CROPS_DIR, d))]

    # B. Vòng lặp xử lý từng ảnh
    print(f"\nBắt đầu xử lý {len(img_folders)} ảnh...")
    
    for img_name in tqdm(img_folders):
        # --- BƯỚC 1: DỰ ĐOÁN (PREDICTION) ---
        # Thay vì ghi ra JSON, ta lưu kết quả vào biến memory_preds
        # memory_preds = { "cell_0": [0,1...], "cell_1": ... }
        memory_preds = {}
        
        crop_folder = os.path.join(INPUT_CROPS_DIR, img_name)
        cell_files = [f for f in os.listdir(crop_folder) if f.endswith('.jpg') and f.startswith('cell_')]
        
        if not cell_files:
            continue

        # Chạy AI cho từng tế bào trong ảnh này
        with torch.no_grad():
            for cell_file in cell_files:
                try:
                    # Parse ID
                    cell_key = os.path.splitext(cell_file)[0] # cell_0
                    
                    # Load & Predict
                    img_path = os.path.join(crop_folder, cell_file)
                    image = Image.open(img_path).convert("RGB")
                    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
                    
                    logits = model(input_tensor)
                    probs = torch.sigmoid(logits)
                    binary_preds = (probs > thresholds).int().cpu().tolist()[0]
                    
                    memory_preds[cell_key] = binary_preds
                except Exception as e:
                    print(f"Err pred {cell_file}: {e}")

        # --- BƯỚC 2: TRỰC QUAN HÓA (VISUALIZATION) ---
        # Lấy ngay kết quả từ memory_preds để vẽ
        
        # Tìm file gốc
        origin_img_path, contour_path = find_files(img_name)
        if not origin_img_path or not contour_path:
            print(f"Thiếu ảnh gốc hoặc contour cho {img_name}. Bỏ qua vẽ.")
            continue
            
        # Load ảnh gốc
        vis_img = cv2.imread(origin_img_path)
        
        # Load contour
        try:
            with open(contour_path, 'r') as f:
                contours_data = json.load(f)
                if isinstance(contours_data, dict): 
                    contours_data = list(contours_data.values())
        except:
            continue

        stats = {'High': 0, 'Medium': 0, 'Low': 0}

        # Vẽ
        for cell_item in contours_data:
            cell_id = cell_item.get('id')
            cell_key = f"cell_{cell_id}"
            
            # Nếu tế bào này không có trong kết quả dự đoán (do lỗi crop hoặc gì đó), bỏ qua
            if cell_key not in memory_preds:
                continue
                
            points = cell_item.get('points') or cell_item.get('contour')
            if not points: continue
            
            cnt = np.array(points, dtype=np.int32)
            pred_vector = memory_preds[cell_key]
            
            # Tính điểm & Màu
            score = np.dot(pred_vector, FEATURE_WEIGHTS)
            color, risk_level = get_risk_color(pred_vector, score)
            
            # Vẽ contour
            cv2.drawContours(vis_img, [cnt], -1, color, THICKNESS)
            stats[risk_level] += 1

        # Lưu ảnh cuối cùng
        out_file = os.path.join(OUTPUT_DIR, f"Vis_{img_name}_Final.jpg")
        cv2.imwrite(out_file, vis_img)
        
        # In báo cáo nhanh
        print(f"\n[Xong {img_name}] -> {out_file}")
        print(f"   Thống kê: {stats['High']} Đỏ | {stats['Medium']} Cam | {stats['Low']} Xanh")

    print("\n=== HOÀN TẤT TOÀN BỘ PIPELINE ===")

if __name__ == "__main__":
    run_pipeline()