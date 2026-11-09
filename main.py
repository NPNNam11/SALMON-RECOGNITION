import cv2
import numpy as np
import sys
import os

# ==========================================
# 1. KHỞI TẠO VÀ ĐỌC ẢNH
# ==========================================
path = sys.argv[1] if len(sys.argv) > 1 else "salmon.jpg"
stem = os.path.splitext(os.path.basename(path))[0]

img = cv2.imread(path)
if img is None:
    raise FileNotFoundError(f"Khong tim thay {path}")

w = 900
h = int(img.shape[0] * w / img.shape[1])
img = cv2.resize(img, (w, h))
height, width = img.shape[:2]


# ==========================================
# HÀM TRÍCH XUẤT TRỤC XƯƠNG CHUẨN (PCA MIDLINE)
# ==========================================
def smooth_curve(pts, win=11):
    """Làm mịn đường cong bằng bộ lọc trung bình động"""
    if len(pts) <= win:
        return pts
    kernel = np.ones(win) / win
    pad = win // 2
    x_pad = np.pad(pts[:, 0], (pad, pad), mode='edge')
    y_pad = np.pad(pts[:, 1], (pad, pad), mode='edge')
    sx = np.convolve(x_pad, kernel, mode='valid')
    sy = np.convolve(y_pad, kernel, mode='valid')
    return np.column_stack((sx, sy))


def get_fish_spine_pca(binary_mask, num_slices=120):
    """
    Xác định trục sống lưng duy nhất từ đầu đến đuôi bằng PCA & Slicing
    """
    pts = np.argwhere(binary_mask > 0)  # (y, x)
    if len(pts) == 0:
        return np.zeros_like(binary_mask), np.array([])

    # Chuyển tọa độ sang dạng (x, y)
    pts_xy = pts[:, [1, 0]].astype(np.float32)

    # 1. Tìm hướng chính của thân cá qua PCA
    mean, eigenvectors = cv2.PCACompute(pts_xy, mean=None)
    center = mean[0]            # Tâm thân cá (x_c, y_c)
    axis_major = eigenvectors[0] # Vector dọc theo chiều dài cá
    axis_minor = eigenvectors[1] # Vector vuông góc (ngang thân cá)

    # 2. Chiếu các điểm lên 2 trục
    vec_pos = pts_xy - center
    proj_major = np.dot(vec_pos, axis_major)
    proj_minor = np.dot(vec_pos, axis_minor)

    # 3. Quét các lát cắt từ đầu đến đuôi
    min_m, max_m = np.min(proj_major), np.max(proj_major)
    
    # Cắt tỉa 3% ở hai đầu mõm/đuôi để loại bỏ phần đỉnh quá nhọn
    margin = (max_m - min_m) * 0.03
    slice_positions = np.linspace(min_m + margin, max_m - margin, num_slices)
    step = (max_m - min_m) / num_slices

    spine_pts = []
    for pos in slice_positions:
        # Lấy các điểm mask nằm trong lát cắt hiện tại
        mask_slice = (proj_major >= pos - step / 2) & (proj_major <= pos + step / 2)
        if np.any(mask_slice):
            # Tìm trung điểm lòng cá tại lát cắt này
            slice_minors = proj_minor[mask_slice]
            mid_minor = np.median(slice_minors)
            
            # Tái tạo tọa độ (x, y) thực tế
            pt_real = center + pos * axis_major + mid_minor * axis_minor
            spine_pts.append(pt_real)

    if len(spine_pts) < 3:
        return np.zeros_like(binary_mask), np.array([])

    spine_pts = np.array(spine_pts)

    # 4. Làm mịn đường trục xương
    spine_pts = smooth_curve(spine_pts, win=15)
    spine_pts_int = spine_pts.astype(np.int32)

    # 5. Vẽ trục xương ra mask nhị phân
    skel_mask = np.zeros_like(binary_mask)
    cv2.polylines(skel_mask, [spine_pts_int], isClosed=False, color=255, thickness=2)

    return skel_mask, spine_pts_int


# ==========================================
# 2. CHỌN ROI
# ==========================================
r = cv2.selectROI("Chon khung quanh ca", img, showCrosshair=False)
cv2.destroyWindow("Chon khung quanh ca")

x, y, rw, rh = r
if rw == 0 or rh == 0:
    raise ValueError("Ban chua chon ROI")

roi_mask = np.zeros((height, width), np.uint8)
roi_mask[y:y+rh, x:x+rw] = 255


# ==========================================
# 3. TẠO MASK HSV BAN ĐẦU
# ==========================================
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([0, 191, 196])
upper = np.array([18, 255, 255])

hsv_mask = cv2.inRange(hsv, lower, upper)
hsv_mask = cv2.bitwise_and(hsv_mask, roi_mask)

kernel_hsv = np.ones((5, 5), np.uint8)
hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, kernel_hsv)
hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel_hsv)

cnts, _ = cv2.findContours(hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not cnts:
    raise ValueError("HSV khong tim thay vung nao. Hay kiem tra nguong HSV.")

big = max(cnts, key=cv2.contourArea)
sure = np.zeros((height, width), np.uint8)
cv2.drawContours(sure, [big], -1, 255, -1)
sure = cv2.erode(sure, np.ones((15, 15), np.uint8))


# ==========================================
# 4. KHỞI TẠO GRABCUT MASK & ĐÁNH DẤU POLYGON
# ==========================================
def create_grabcut_mask():
    mask = np.full((height, width), cv2.GC_BGD, np.uint8)
    mask[roi_mask > 0] = cv2.GC_PR_BGD

    near = cv2.dilate(sure, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    near = cv2.bitwise_and(near, roi_mask)

    mask[near > 0] = cv2.GC_PR_FGD
    mask[sure > 0] = cv2.GC_FGD
    mask[seed_mask == 1] = cv2.GC_FGD
    mask[seed_mask == 2] = cv2.GC_BGD

    return mask

seed_mask = np.zeros((height, width), np.uint8)
polygons = []
current_pts = []
mode = 1
COLORS = {1: (0, 255, 0), 2: (0, 0, 255)}

def rebuild_seed():
    seed_mask[:] = 0
    for m, pts in polygons:
        cv2.fillPoly(seed_mask, [pts], int(m))
    seed_mask[roi_mask == 0] = 0

def close_polygon():
    global current_pts
    if len(current_pts) >= 3:
        polygons.append((mode, np.array(current_pts, np.int32)))
        rebuild_seed()
    else:
        print("Can it nhat 3 diem de tao polygon")
    current_pts = []

def redraw():
    canvas = img.copy()
    overlay = img.copy()
    for m, pts in polygons:
        cv2.fillPoly(overlay, [pts], COLORS[m])
    canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)

    for m, pts in polygons:
        cv2.polylines(canvas, [pts], True, COLORS[m], 2)

    if current_pts:
        pts = np.array(current_pts, np.int32)
        cv2.polylines(canvas, [pts], False, COLORS[mode], 2)
        for p in current_pts:
            cv2.circle(canvas, p, 4, COLORS[mode], -1)

    cv2.rectangle(canvas, (x, y), (x + rw, y + rh), (255, 255, 0), 1)
    text = "CHE DO: CA (1)" if mode == 1 else "CHE DO: NEN (2)"
    cv2.putText(canvas, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[mode], 2)
    return canvas

def mouse(event, mx, my, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if roi_mask[my, mx] > 0:
            current_pts.append((mx, my))
    elif event == cv2.EVENT_RBUTTONDOWN:
        close_polygon()


# ==========================================
# 5. GIAO DIỆN & VÒNG LẬP CHÍNH
# ==========================================
cv2.namedWindow("Danh dau")
cv2.setMouseCallback("Danh dau", mouse)

print("--------------------------------")
print("1          : CHE DO DANH DAU CA (XANH)")
print("2          : CHE DO DANH DAU NEN (DO)")
print("CHUOT TRAI : THEM DINH POLYGON")
print("CHUOT PHAI / ENTER : DONG POLYGON")
print("Z          : BO DINH CUOI (dang ve)")
print("U          : XOA POLYGON DA DONG CUOI CUNG")
print("R          : XOA HET")
print("G          : CHAY GRABCUT & TRÍCH XUẤT TRỤC XƯƠNG")
print("Q          : THOAT")
print("--------------------------------")

while True:
    cv2.imshow("Danh dau", redraw())
    key = cv2.waitKey(20) & 0xFF

    if key == ord('1'):
        mode = 1
    elif key == ord('2'):
        mode = 2
    elif key == 13:  # Enter
        close_polygon()
    elif key == ord('z'):
        if current_pts:
            current_pts.pop()
    elif key == ord('u'):
        if polygons:
            polygons.pop()
            rebuild_seed()
    elif key == ord('r'):
        polygons.clear()
        current_pts.clear()
        rebuild_seed()
    elif key == ord('g'):
        if len(current_pts) >= 3:
            close_polygon()

        # 1. Chạy GrabCut
        mask = create_grabcut_mask()
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(img, mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)

        # 2. Xử lý mask
        mask2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel_smooth)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel_smooth)
        mask_blur = cv2.GaussianBlur(mask2, (7, 7), 1.5)
        _, mask2 = cv2.threshold(mask_blur, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            clean_mask = np.zeros_like(mask2)
            cv2.drawContours(clean_mask, [c], -1, 255, -1)

            perimeter = cv2.arcLength(c, True)
            epsilon = 0.0015 * perimeter
            c_smooth = cv2.approxPolyDP(c, epsilon, True)

            # 3. TRÍCH XUẤT TRỤC XƯƠNG CHUẨN ĐƠN (PCA MIDLINE)
            skel_mask, spine_pts = get_fish_spine_pca(clean_mask)

            # 4. HIỂN THỊ KẾT QUẢ
            out_final = img.copy()

            # Viền thân cá (Xanh lá)
            cv2.drawContours(out_final, [c_smooth], -1, (0, 255, 0), 2)

            # Trục xương sống (Đường Đỏ dày 3px)
            if len(spine_pts) > 0:
                cv2.polylines(out_final, [spine_pts], isClosed=False, color=(0, 0, 255), thickness=3)

            # Ghi chú
            cv2.putText(out_final, "Vien ca: Xanh la", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(out_final, "Truc xuong song: Do", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Mask sach", clean_mask)
            cv2.imshow("Truc Xuong Ca Smooth", out_final)
            cv2.imshow("Mask Truc Xuong", skel_mask)

            # 5. LƯU FILE
            cv2.imwrite(f"{stem}_mask.png", clean_mask)
            cv2.imwrite(f"{stem}_spine_result.png", out_final)
            
            if len(spine_pts) > 0:
                np.savetxt(
                    f"{stem}_spine_pts.csv",
                    spine_pts,
                    delimiter=",",
                    header="x,y",
                    comments="",
                    fmt="%d"
                )
            print(f"Thanh cong! Da luu anh ket qua va toa do trục xương {stem}_spine_pts.csv")
        else:
            print("Khong tim thay contour!")

    elif key == ord('q'):
        break

cv2.destroyAllWindows()
