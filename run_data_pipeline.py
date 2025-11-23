import cv2
import json
import numpy as np
import os
import shutil

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN (PATH CONFIG)
# ==========================================
# Sử dụng đường dẫn tuyệt đối để tránh lỗi
BASE_DIR = r"E:\Work\Github\vnu-tool\Dataset"

INPUT_IMAGES_DIR = os.path.join(BASE_DIR, "images")
INPUT_CONTOUR_DIR = os.path.join(BASE_DIR, "contour")

# Output của Step 1 (Trung gian)
STEP1_OUTPUT_DIR = os.path.join(BASE_DIR, "step1_transparent_crops")

# Output của Step 2 (Kết quả cuối cùng để đưa vào Model)
STEP2_OUTPUT_DIR = os.path.join(BASE_DIR, "step2_model_input_224")

# ==========================================
# 2. CẤU HÌNH THAM SỐ (PARAMS)
# ==========================================
# Step 1: Padding khi cắt contour (pixel)
CROP_PADDING = 1 

# Step 2: Chuẩn hóa
TARGET_SIZE = 224
BACKGROUND_COLOR = (255, 255, 255) # Trắng
CELL_FILL_RATIO = 0.45 # Tế bào chiếm khoảng 45% khung hình

# ==========================================
# 3. HÀM HỖ TRỢ (UTILS)
# ==========================================
def load_json_contours(img_name):
    """Tìm và đọc file json contour."""
    possible_names = [f"{img_name}_contours.json", f"{img_name}.json"]
    
    found_path = None
    for name in possible_names:
        path = os.path.join(INPUT_CONTOUR_DIR, name)
        if os.path.exists(path):
            found_path = path
            break
    
    if not found_path:
        return []

    try:
        with open(found_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, list): return data
        if isinstance(data, dict): return list(data.values())
        return []
    except:
        return []

# ==========================================
# 4. GIAI ĐOẠN 1: CROP TRANSPARENT
# ==========================================
def run_step1_crop_transparent():
    print(f"\n>>> BẮT ĐẦU STEP 1: Cắt ảnh & Tách nền trong suốt...")
    
    # Dọn dẹp thư mục cũ
    if os.path.exists(STEP1_OUTPUT_DIR):
        shutil.rmtree(STEP1_OUTPUT_DIR)
    os.makedirs(STEP1_OUTPUT_DIR, exist_ok=True)

    image_files = [f for f in os.listdir(INPUT_IMAGES_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    total_cells = 0

    for img_file in image_files:
        img_name = os.path.splitext(img_file)[0]
        src_img_path = os.path.join(INPUT_IMAGES_DIR, img_file)
        
        contours = load_json_contours(img_name)
        if not contours:
            continue

        original_img = cv2.imread(src_img_path)
        if original_img is None: continue
        h_img, w_img = original_img.shape[:2]

        # Tạo thư mục con cho ảnh này
        save_dir = os.path.join(STEP1_OUTPUT_DIR, img_name)
        os.makedirs(save_dir, exist_ok=True)

        count = 0
        for item in contours:
            pts = item.get('points') or item.get('contour')
            cell_id = item.get('id', count)
            if not pts: continue

            cnt = np.array(pts, dtype=np.int32)
            
            # 1. Bounding Rect
            x, y, w, h = cv2.boundingRect(cnt)
            
            # 2. Padding an toàn
            x1, y1 = max(0, x - CROP_PADDING), max(0, y - CROP_PADDING)
            x2, y2 = min(w_img, x + w + CROP_PADDING), min(h_img, y + h + CROP_PADDING)
            
            cell_roi = original_img[y1:y2, x1:x2].copy()
            
            # 3. Masking (Tạo kênh Alpha)
            mask = np.zeros(cell_roi.shape[:2], dtype=np.uint8)
            roi_cnt = cnt - [x1, y1]
            cv2.drawContours(mask, [roi_cnt], -1, (255), thickness=cv2.FILLED)
            
            b, g, r = cv2.split(cell_roi)
            rgba = cv2.merge((b, g, r, mask)) # Ảnh 4 kênh
            
            # Lưu PNG
            cv2.imwrite(os.path.join(save_dir, f"cell_{cell_id}.png"), rgba)
            count += 1
        
        total_cells += count
        print(f"   Processed {img_name}: {count} cells")

    print(f"[XONG STEP 1] Tổng cộng {total_cells} tế bào trong suốt được tạo tại: {STEP1_OUTPUT_DIR}")

# ==========================================
# 5. GIAI ĐOẠN 2: NORMALIZE (RESIZE & CENTER)
# ==========================================
def normalize_single_cell(rgba_img):
    h, w = rgba_img.shape[:2]
    
    # Tính tỷ lệ Scale theo cấu hình CELL_FILL_RATIO (0.45)
    target_dim = int(TARGET_SIZE * CELL_FILL_RATIO)
    scale = target_dim / max(h, w)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize (Upscale dùng Cubic, Downscale dùng Area)
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    resized = cv2.resize(rgba_img, (new_w, new_h), interpolation=interp)
    
    # Tạo Canvas trắng
    canvas = np.ones((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
    canvas[:] = BACKGROUND_COLOR
    
    # Tính vị trí trung tâm
    x_center = (TARGET_SIZE - new_w) // 2
    y_center = (TARGET_SIZE - new_h) // 2
    
    # Alpha Blending
    cell_rgb = resized[:, :, :3]
    alpha = resized[:, :, 3].astype(float) / 255.0
    
    roi = canvas[y_center:y_center+new_h, x_center:x_center+new_w]
    for c in range(3):
        roi[:, :, c] = (alpha * cell_rgb[:, :, c] + (1.0 - alpha) * roi[:, :, c])
        
    canvas[y_center:y_center+new_h, x_center:x_center+new_w] = roi
    return canvas

def run_step2_normalize():
    print(f"\n>>> BẮT ĐẦU STEP 2: Chuẩn hóa về 224x224 (Ratio {CELL_FILL_RATIO})...")
    
    if os.path.exists(STEP2_OUTPUT_DIR):
        shutil.rmtree(STEP2_OUTPUT_DIR)
    os.makedirs(STEP2_OUTPUT_DIR, exist_ok=True)
    
    # Duyệt qua các thư mục trong Step 1
    img_dirs = [d for d in os.listdir(STEP1_OUTPUT_DIR) if os.path.isdir(os.path.join(STEP1_OUTPUT_DIR, d))]
    
    for img_name in img_dirs:
        src_dir = os.path.join(STEP1_OUTPUT_DIR, img_name)
        dest_dir = os.path.join(STEP2_OUTPUT_DIR, img_name)
        os.makedirs(dest_dir, exist_ok=True)
        
        png_files = [f for f in os.listdir(src_dir) if f.endswith('.png')]
        
        for f in png_files:
            rgba_path = os.path.join(src_dir, f)
            rgba_img = cv2.imread(rgba_path, cv2.IMREAD_UNCHANGED)
            
            if rgba_img is None: continue
            
            # Chuẩn hóa
            final_img = normalize_single_cell(rgba_img)
            
            # Lưu JPG (cho Model)
            save_name = f.replace('.png', '.jpg')
            cv2.imwrite(os.path.join(dest_dir, save_name), final_img)
            
        print(f"   Normalized {img_name}: {len(png_files)} cells")

    print(f"[XONG STEP 2] Dữ liệu Model Input sẵn sàng tại: {STEP2_OUTPUT_DIR}")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Chạy tuần tự
    run_step1_crop_transparent()
    run_step2_normalize()
    print("\n=== DATA PIPELINE HOÀN TẤT ===")