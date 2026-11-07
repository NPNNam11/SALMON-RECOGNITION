
import cv2
import numpy as np

# ==========================================
# 1. DOC ANH VA THU NHO
# ==========================================

img = cv2.imread("salmon.jpg")

if img is None:
    raise FileNotFoundError("Khong tim thay salmon.jpg")

w = 900
h = int(img.shape[0] * w / img.shape[1])
img = cv2.resize(img, (w, h))

height, width = img.shape[:2]


# ==========================================
# 2. CHON ROI
# ==========================================

r = cv2.selectROI(
    "Chon khung quanh ca",
    img,
    showCrosshair=False
)

cv2.destroyWindow("Chon khung quanh ca")

x, y, rw, rh = r

if rw == 0 or rh == 0:
    raise ValueError("Ban chua chon ROI")


# Tao mask ROI
roi_mask = np.zeros((height, width), np.uint8)
roi_mask[y:y+rh, x:x+rw] = 255


# ==========================================
# 3. TAO MASK HSV
# ==========================================

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower = np.array([0, 191, 196])
upper = np.array([18, 255, 255])

hsv_mask = cv2.inRange(hsv, lower, upper)

# Chi xet HSV trong ROI
hsv_mask = cv2.bitwise_and(hsv_mask, roi_mask)

kernel_hsv = np.ones((5, 5), np.uint8)

hsv_mask = cv2.morphologyEx(
    hsv_mask,
    cv2.MORPH_CLOSE,
    kernel_hsv
)

hsv_mask = cv2.morphologyEx(
    hsv_mask,
    cv2.MORPH_OPEN,
    kernel_hsv
)


# Tim vung HSV lon nhat
cnts, _ = cv2.findContours(
    hsv_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if not cnts:
    raise ValueError("HSV khong tim thay vung nao. Hay kiem tra nguong HSV.")

big = max(cnts, key=cv2.contourArea)

sure = np.zeros((height, width), np.uint8)

cv2.drawContours(
    sure,
    [big],
    -1,
    255,
    -1
)

# Erosion: lay vung chac chan la ca
sure = cv2.erode(
    sure,
    np.ones((15, 15), np.uint8)
)


# ==========================================
# 4. KHOI TAO GRABCUT MASK
# ==========================================

def create_grabcut_mask():

    mask = np.full(
        (height, width),
        cv2.GC_BGD,
        np.uint8
    )

    # Trong ROI: co the la nen
    mask[roi_mask > 0] = cv2.GC_PR_BGD

    # Mo rong vung co kha nang la ca
    near = cv2.dilate(
        sure,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (31, 31)
        )
    )

    near = cv2.bitwise_and(near, roi_mask)

    mask[near > 0] = cv2.GC_PR_FGD

    # Vung HSV chac chan la ca
    mask[sure > 0] = cv2.GC_FGD

    # Ap dung danh dau thu cong
    mask[seed_mask == 1] = cv2.GC_FGD
    mask[seed_mask == 2] = cv2.GC_BGD

    return mask


# ==========================================
# 5. DANH DAU THU CONG
# ==========================================

# 0 = chua danh dau
# 1 = ca
# 2 = nen

seed_mask = np.zeros(
    (height, width),
    np.uint8
)

canvas = img.copy()

drawing = False
mode = None

brush = 7


def mouse(event, mx, my, flags, param):

    global drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        mode = 1

    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing = True
        mode = 2

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        mode = None

    elif event == cv2.EVENT_RBUTTONUP:
        drawing = False
        mode = None

    if drawing and mode is not None:

        if 0 <= mx < width and 0 <= my < height:

            if roi_mask[my, mx] > 0:

                cv2.circle(
                    seed_mask,
                    (mx, my),
                    brush,
                    mode,
                    -1
                )

                color = (
                    (0, 255, 0)
                    if mode == 1
                    else (0, 0, 255)
                )

                cv2.circle(
                    canvas,
                    (mx, my),
                    brush,
                    color,
                    -1
                )


# ==========================================
# 6. GIAO DIEN
# ==========================================

cv2.namedWindow("Danh dau")
cv2.setMouseCallback("Danh dau", mouse)

print("--------------------------------")
print("CHUOT TRAI : DANH DAU CA (XANH)")
print("CHUOT PHAI : DANH DAU NEN (DO)")
print("G           : CHAY GRABCUT")
print("Q           : THOAT")
print("--------------------------------")


# ==========================================
# 7. VONG LAP CHINH
# ==========================================

while True:

    cv2.imshow("Danh dau", canvas)

    key = cv2.waitKey(20) & 0xFF

    if key == ord('g'):

        # Tao mask GrabCut moi
        mask = create_grabcut_mask()

        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)

        # Chay GrabCut
        cv2.grabCut(
            img,
            mask,
            None,
            bgd,
            fgd,
            5,
            cv2.GC_INIT_WITH_MASK
        )

        # ==================================
        # 8. TAO MASK NHI PHAN
        # ==================================

        mask2 = np.where(
            (mask == cv2.GC_FGD) |
            (mask == cv2.GC_PR_FGD),
            255,
            0
        ).astype(np.uint8)


        # ==================================
        # 9. LOC NHIỄU VA LAM MIN MASK
        # ==================================

        kernel_smooth = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        )

        # Loai bo nhieu nho
        mask2 = cv2.morphologyEx(
            mask2,
            cv2.MORPH_OPEN,
            kernel_smooth
        )

        # Lap cac khe ho nho
        mask2 = cv2.morphologyEx(
            mask2,
            cv2.MORPH_CLOSE,
            kernel_smooth
        )

        # Gaussian Blur
        mask_blur = cv2.GaussianBlur(
            mask2,
            (7, 7),
            1.5
        )

        # Chuyen lai thanh mask nhi phan
        _, mask2 = cv2.threshold(
            mask_blur,
            127,
            255,
            cv2.THRESH_BINARY
        )


        # ==================================
        # 10. TIM CONTOUR
        # ==================================

        contours, _ = cv2.findContours(
            mask2,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        if contours:

            # Lay contour lon nhat
            c = max(
                contours,
                key=cv2.contourArea
            )

            # Tao mask sach chi chua contour lon nhat
            clean_mask = np.zeros_like(mask2)

            cv2.drawContours(
                clean_mask,
                [c],
                -1,
                255,
                -1
            )


            # ==================================
            # 11. DON GIAN HOA CONTOUR
            # ==================================

            perimeter = cv2.arcLength(c, True)

            epsilon = 0.0015 * perimeter

            c_smooth = cv2.approxPolyDP(
                c,
                epsilon,
                True
            )


            # ==================================
            # 12. HIEN THI KET QUA
            # ==================================

            out = img.copy()

            cv2.drawContours(
                out,
                [c_smooth],
                -1,
                (0, 255, 0),
                2
            )

            cv2.imshow(
                "Mask GrabCut",
                mask2
            )

            cv2.imshow(
                "Mask sach",
                clean_mask
            )

            cv2.imshow(
                "Result - Contour",
                out
            )

            print("Dien tich contour:", cv2.contourArea(c))
            print("So diem contour ban dau:", len(c))
            print("So diem sau don gian hoa:", len(c_smooth))

            # Luu mask va contour
            cv2.imwrite("salmon_mask.png", clean_mask)

            cv2.imwrite(
                "salmon_contour.png",
                out
            )

            np.save(
                "salmon_contour.npy",
                c_smooth
            )

        else:
            print("Khong tim thay contour!")

    elif y == ord('q'):
        break


cv2.destroyAllWindows()
