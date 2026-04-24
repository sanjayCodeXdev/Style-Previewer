"""
STYLE Previewer
---------------
An AR virtual try-on system for eyebrows.
Features:
- Landmark tracking and smoothing
- Color and lighting matching
- Automatic style recommendation

Authors: Sanjay, Subasri & Srimathi
"""

import cv2
import mediapipe as mp
import numpy as np
import os

# --- Initialization ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# Landmarks tracking points
LEFT_EYEBROW = [107, 66, 105, 63, 70, 46, 53, 52, 65, 55] 
RIGHT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
ALL_EYEBROWS = LEFT_EYEBROW + RIGHT_EYEBROW

# Load assets
assets_dir = r'D:\OneDrive\Documents\STYLE_Previewer\assets\Eyebrows'
eyebrow_sheets = []
if os.path.exists(assets_dir):
    eyebrow_sheets = [os.path.join(assets_dir, f) for f in os.listdir(assets_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

eyebrow_styles = []
eyebrow_scale = 1.0

def crop_alpha_region(img):
    alpha_channel = img[:, :, 3]
    contours, _ = cv2.findContours(alpha_channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    
    valid_contours = [c for c in contours if cv2.contourArea(c) > 50]
    if not valid_contours:
        valid_contours = [max(contours, key=cv2.contourArea)]
        
    pts = np.vstack(valid_contours)
    x, y, w, h = cv2.boundingRect(pts)
    x, y = max(0, x - 1), max(0, y - 1)
    w, h = min(img.shape[1] - x, w + 2), min(img.shape[0] - y, h + 2)
    return img[y:y+h, x:x+w]

for sheet_path in eyebrow_sheets:
    sheet = cv2.imread(sheet_path, cv2.IMREAD_UNCHANGED)
    if sheet is not None:
        num_styles = 5
        h, w = sheet.shape[:2]
        for i in range(num_styles):
            y_start, y_end = i * (h // num_styles), (i + 1) * (h // num_styles)
            bgr = sheet[y_start:y_end, :, :3] 
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            _, alpha = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
            b, g, r = cv2.split(bgr)
            rgba = cv2.merge((b, g, r, alpha))
            mid_x = w // 2
            left_half, right_half = rgba[:, :mid_x], rgba[:, mid_x:]
            eyebrow_styles.append((crop_alpha_region(left_half), crop_alpha_region(right_half)))

class LandmarkSmoother:
    def __init__(self, slow_alpha=0.2, fast_alpha=0.6, threshold=0.015):
        self.prev_pos = None
        self.smoothed_pos = None
        self.slow_alpha = slow_alpha
        self.fast_alpha = fast_alpha
        self.threshold = threshold

    def smooth(self, current_pos):
        if self.prev_pos is None:
            self.prev_pos = current_pos
            self.smoothed_pos = current_pos
            return current_pos
        
        movement = np.linalg.norm(np.array(current_pos) - np.array(self.prev_pos))
        alpha = self.fast_alpha if movement > self.threshold else self.slow_alpha
        
        self.smoothed_pos = (
            alpha * current_pos[0] + (1 - alpha) * self.smoothed_pos[0],
            alpha * current_pos[1] + (1 - alpha) * self.smoothed_pos[1]
        )
        
        self.prev_pos = current_pos
        return self.smoothed_pos

smoothers = {i: LandmarkSmoother() for i in ALL_EYEBROWS}

class StyleRecommender:
    def __init__(self):
        self.frame_count = 0
        self.interval = 30
        
        self.suitability = {
            "Round":  [10, 6, 4, 3, 2],
            "Square": [3, 4, 6, 10, 8],
            "Long":   [2, 10, 8, 4, 5],
            "Oval":   [7, 8, 9, 10, 9],
            "Heart":  [4, 5, 8, 7, 10]
        }
        self.spacing_mods = {
            "Wide-set": [-2, 2, 1, 0, 0],
            "Close-set": [2, -2, 0, 1, 1],
            "Proportional": [0, 0, 0, 0, 0]
        }

    def analyze_face(self, landmarks, w, h):
        face_h = np.hypot(landmarks[10].x - landmarks[152].x, landmarks[10].y - landmarks[152].y) * h
        face_w = np.hypot(landmarks[234].x - landmarks[454].x, landmarks[234].y - landmarks[454].y) * w
        jaw_w = np.hypot(landmarks[172].x - landmarks[397].x, landmarks[172].y - landmarks[397].y) * w
        ratio = face_h / face_w
        
        shape = "Oval" 
        if ratio < 1.1:
            shape = "Square" if jaw_w > face_w * 0.85 else "Round"
        elif ratio > 1.4: 
            shape = "Long"
        
        inter_ocular = np.hypot(landmarks[133].x - landmarks[362].x, landmarks[133].y - landmarks[362].y) * w
        eye_w = np.hypot(landmarks[33].x - landmarks[133].x, landmarks[33].y - landmarks[133].y) * w
        
        spacing = "Proportional"
        if inter_ocular > eye_w * 1.2: spacing = "Wide-set"
        elif inter_ocular < eye_w * 0.8: spacing = "Close-set"
        
        return shape, spacing

    def get_recommendation(self, shape, spacing, num_styles):
        scores = self.suitability.get(shape, [5]*5)
        mods = self.spacing_mods.get(spacing, [0]*5)
        
        final_scores = []
        for i in range(min(num_styles, len(scores))):
            score = scores[i] + mods[i] + np.random.uniform(-0.5, 0.5)
            final_scores.append(score)
            
        return int(np.argmax(final_scores))

    def sample_color(self, frame, landmarks, indices, w, h):
        points = []
        for i in indices:
            px = int(np.clip(landmarks[i].x * w, 0, w - 1))
            py = int(np.clip(landmarks[i].y * h, 0, h - 1))
            points.append(frame[py, px])
        if not points: return (30, 30, 30)
        return np.median(points, axis=0).tolist()

def apply_tint(asset, color, lighting=1.0):
    b, g, r, a = cv2.split(asset)
    gray = cv2.cvtColor(cv2.merge((b, g, r)), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    l_m = 0.4 + 0.6 * lighting
    t_b, t_g, t_r = color[0] * l_m, color[1] * l_m, color[2] * l_m
    new_b = (t_b * (1.0 - gray) + gray * b).astype(np.uint8)
    new_g = (t_g * (1.0 - gray) + gray * g).astype(np.uint8)
    new_r = (t_r * (1.0 - gray) + gray * r).astype(np.uint8)
    return cv2.merge((new_b, new_g, new_r, a))

def match_lighting(source, target_roi):
    if target_roi is None or target_roi.size == 0 or target_roi.shape[0] < 5 or target_roi.shape[1] < 5: 
        return source
    
    try:
        s_bgr = source[:, :, :3]
        s_gray = cv2.cvtColor(s_bgr, cv2.COLOR_BGR2GRAY)
        t_gray = cv2.cvtColor(target_roi, cv2.COLOR_BGR2GRAY)
        
        t_hist, _ = np.histogram(t_gray.flatten(), 256, [0, 256])
        t_cdf = t_hist.cumsum()
        t_cdf_n = t_cdf * 255 / t_cdf[-1]
        
        s_hist, _ = np.histogram(s_gray.flatten(), 256, [0, 256])
        s_cdf = s_hist.cumsum()
        s_cdf_n = s_cdf * 255 / s_cdf[-1]
        
        lut = np.zeros(256, dtype=np.uint8)
        j = 0
        for i in range(256):
            while j < 255 and t_cdf_n[j] < s_cdf_n[i]:
                j += 1
            lut[i] = j
            
        res_bgr = cv2.LUT(s_bgr, lut)
        return cv2.merge((res_bgr[:, :, 0], res_bgr[:, :, 1], res_bgr[:, :, 2], source[:, :, 3]))
    except Exception:
        return source

recommender = StyleRecommender()
hair_color = (30, 30, 30)
ambient_light = 1.0
roi = None
recommended_idx = 0
analysis_msg = "Analyzing..."
auto_mode = True 
current_idx = 0
asset_cache = {}

clicked_btn = None
def mouse_handler(event, x, y, flags, param):
    global clicked_btn, auto_mode, current_idx
    if event == cv2.EVENT_LBUTTONDOWN:
        if 20 <= x <= 120 and 80 <= y <= 130: clicked_btn = "PREV"
        elif 140 <= x <= 240 and 80 <= y <= 130: clicked_btn = "NEXT"
        elif 260 <= x <= 360 and 80 <= y <= 130: clicked_btn = "SHOT"
        elif 380 <= x <= 480 and 80 <= y <= 130: clicked_btn = "AI"

window_name = "STYLE Previewer"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_handler)

def update_scale(val):
    global eyebrow_scale
    eyebrow_scale = val / 100.0

cv2.createTrackbar("Scale %", window_name, 100, 200, update_scale)

cap = cv2.VideoCapture(0)
if len(eyebrow_styles) == 0:
    print("Warning: No eyebrow assets found!")
    if cap.isOpened(): cap.release()
    cv2.destroyAllWindows()
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: continue

    frame = cv2.flip(frame, 1)
    h_fp, w_fp, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            lms = face_landmarks.landmark
            
            if recommender.frame_count % recommender.interval == 0:
                if auto_mode:
                    shape, spacing = recommender.analyze_face(lms, w_fp, h_fp)
                    recommended_idx = recommender.get_recommendation(shape, spacing, len(eyebrow_styles))
                    current_idx = recommended_idx
                    analysis_msg = f"{shape} Face -> Best Style ({spacing})"
                
                hair_color = recommender.sample_color(frame, lms, ALL_EYEBROWS, w_fp, h_fp)
                
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                samples = [gray_frame[int(np.clip(lms[i].y * h_fp, 0, h_fp - 1)), int(np.clip(lms[i].x * w_fp, 0, w_fp - 1))] for i in [8, 234, 454, 152]]
                ambient_light = np.mean(samples) / 255.0
                
                fh = [lms[i] for i in [10, 8, 338, 109]]
                fx1, fx2 = int(min(p.x for p in fh) * w_fp), int(max(p.x for p in fh) * w_fp)
                fy1, fy2 = int(min(p.y for p in fh) * h_fp), int(max(p.y for p in fh) * h_fp)
                roi = frame[max(0, fy1):min(h_fp, fy2), max(0, fx1):min(w_fp, fx2)]
            
            recommender.frame_count += 1
            
            smoothed_pts = {i: smoothers[i].smooth((lms[i].x, lms[i].y)) for i in ALL_EYEBROWS}
            
            cache_key = (current_idx, tuple(map(int, hair_color)), int(ambient_light * 100))
            if cache_key in asset_cache:
                left_final, right_final = asset_cache[cache_key]
            else:
                left_orig, right_orig = eyebrow_styles[current_idx]
                l_tint = apply_tint(left_orig, hair_color, ambient_light)
                r_tint = apply_tint(right_orig, hair_color, ambient_light)
                left_final = match_lighting(l_tint, roi)
                right_final = match_lighting(r_tint, roi)
                asset_cache[cache_key] = (left_final, right_final)
                if len(asset_cache) > 100: asset_cache.clear()

            def overlay_asset(indices, asset):
                xs = [int(smoothed_pts[i][0] * w_fp) for i in indices]
                ys = [int(smoothed_pts[i][1] * h_fp) for i in indices]
                
                try:
                    coeffs = np.polyfit(xs, ys, 2)
                    curv_boost = 1.0 + abs(coeffs[0]) * 1000.0
                except:
                    curv_boost = 1.0
                
                min_x_idx, max_x_idx = xs.index(min(xs)), xs.index(max(xs))
                dx, dy = xs[max_x_idx] - xs[min_x_idx], ys[max_x_idx] - ys[min_x_idx]
                length = np.hypot(dx, dy)
                if length < 5: return
                
                angle = np.degrees(np.arctan2(dy, dx))
                nx, ny = -dy / length, dx / length
                projs = [xs[j] * nx + ys[j] * ny for j in range(len(xs))]
                thickness = max(projs) - min(projs)
                
                req_w = int(max(10.0, length * 1.05 * eyebrow_scale))
                req_h = int(max(10.0, thickness * 1.5 * eyebrow_scale * min(1.3, curv_boost)))
                
                resized = cv2.resize(asset, (req_w, req_h))
                b, g, r, a = cv2.split(resized)
                a = cv2.GaussianBlur(a, (3, 3), 0)
                noise = np.random.normal(0, 3, (req_h, req_w, 3)).astype(np.float32)
                rgb = cv2.merge((b, g, r)).astype(np.float32)
                rgb = np.clip(cv2.add(rgb, noise), 0, 255).astype(np.uint8)
                resized = cv2.merge((cv2.split(rgb)[0], cv2.split(rgb)[1], cv2.split(rgb)[2], a))

                center_a, center_f = (req_w / 2.0, req_h / 2.0), (np.mean(xs), np.mean(ys))
                
                M = cv2.getRotationMatrix2D(center_a, angle, 1.0)
                M[0, 2] += center_f[0] - center_a[0]
                M[1, 2] += center_f[1] - center_a[1]
                warped = cv2.warpAffine(resized, M, (w_fp, h_fp), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
                
                mask = warped[:, :, 3] / 255.0
                img_rgb = warped[:, :, :3].astype(np.float32)
                alpha_expanded = mask[:, :, np.newaxis]
                frame[:] = ((alpha_expanded * img_rgb) + ((1 - alpha_expanded) * frame.astype(np.float32))).astype(np.uint8)

            l_mean_x = np.mean([smoothed_pts[i][0] for i in LEFT_EYEBROW])
            r_mean_x = np.mean([smoothed_pts[i][0] for i in RIGHT_EYEBROW])
            if l_mean_x < r_mean_x:
                overlay_asset(LEFT_EYEBROW, left_final)
                overlay_asset(RIGHT_EYEBROW, right_final)
            else:
                overlay_asset(LEFT_EYEBROW, right_final)
                overlay_asset(RIGHT_EYEBROW, left_final)

    # UI Overlay
    disp = frame.copy()
    ui_layer = disp.copy()
    cv2.rectangle(ui_layer, (0, 0), (w_fp, 150), (0, 0, 0), -1)
    if auto_mode:
        cv2.rectangle(ui_layer, (0, h_fp - 40), (w_fp, h_fp), (40, 40, 0), -1)
    
    cv2.addWeighted(ui_layer, 0.4, disp, 0.6, 0, disp)
    
    def draw_button(img, label, x_bounds, color):
        cv2.rectangle(img, (x_bounds[0], 80), (x_bounds[1], 130), color, -1)
        cv2.putText(img, label, (x_bounds[0] + 15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    draw_button(disp, "PREV", (20, 120), (50, 50, 50))
    draw_button(disp, "NEXT", (140, 240), (50, 50, 50))
    draw_button(disp, "SHOT", (260, 360), (0, 80, 0))
    draw_button(disp, "AUTO" if auto_mode else "MANUAL", (380, 500), (80, 0, 0))
    
    cv2.putText(disp, f"STYLE {current_idx + 1}/{len(eyebrow_styles)}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(disp, f"Status: {analysis_msg}", (220, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    status_text = "AUTO SELECTION ACTIVE" if auto_mode else "MANUAL SELECTION"
    status_color = (0, 255, 255) if auto_mode else (100, 100, 255)
    cv2.putText(disp, status_text, (20, h_fp - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

    if clicked_btn:
        if clicked_btn == "PREV":
            auto_mode = False
            current_idx = (current_idx - 1) % len(eyebrow_styles)
        elif clicked_btn == "NEXT":
            auto_mode = False
            current_idx = (current_idx + 1) % len(eyebrow_styles)
        elif clicked_btn == "AI":
            auto_mode = not auto_mode
        elif clicked_btn == "SHOT":
            fn = os.path.join(os.getcwd(), f"snapshot_{current_idx}.png")
            cv2.imwrite(fn, frame) 
            print(f"Saved snapshot to {fn}")
        clicked_btn = None
        
    cv2.imshow(window_name, disp)

    key = cv2.waitKeyEx(5)
    if key == 27:
        break
    elif key in (ord('a'), ord('A')):
        auto_mode = not auto_mode
        analysis_msg = "Recalculating..." if auto_mode else "Manual Mode"  
    elif key in (ord('w'), ord('W'), 2490368):
        auto_mode = False
        analysis_msg = "Manual Mode"
        if eyebrow_styles:
            current_idx = (current_idx - 1) % len(eyebrow_styles)
    elif key in (ord('s'), ord('S'), 2621440):
        auto_mode = False
        analysis_msg = "Manual Mode"
        if eyebrow_styles:
            current_idx = (current_idx + 1) % len(eyebrow_styles)
    elif key in (ord('+'), ord('=')):
        eyebrow_scale += 0.05
    elif key in (ord('-'), ord('_')):
        eyebrow_scale = max(0.1, eyebrow_scale - 0.05)

cap.release()
cv2.destroyAllWindows()