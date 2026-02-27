import numpy as np
import cv2
import os
from scipy.optimize import minimize

# [Source: 11, 24] 使用 D65 光源下的 sRGB 數值
TARGET_SRGB = np.array([
    [115, 82, 68], # 1. dark skin
    [194, 150, 130], # 2. light skin
    [98, 122, 157], # 3. blue sky
    [87, 108, 67], # 4. foliage
    [133, 128, 177], # 5. blue flower
    [103, 189, 170], # 6. bluish green
    [214, 126, 44], # 7. orange
    [80, 91, 166], # 8. purplish blue
    [193, 90, 99], # 9. moderate red
    [94, 60, 108], # 10. purple
    [157, 188, 64], # 11. yellow green
    [224, 163, 46], # 12. orange yellow
    [56, 61, 150], # 13. blue
    [70, 148, 73], # 14. green
    [175, 54, 60], # 15. red
    [231, 199, 31], # 16. yellow
    [187, 86, 149], # 17. magenta
    [8, 133, 161], # 18. cyan
    [243, 243, 242], # 19. white
    [200, 200, 200], # 20. neutral 8
    [160, 160, 160], # 21. neutral 6.5
    [122, 122, 121], # 22. neutral 5
    [85, 85, 85], # 23. neutral 3.5
    [52, 52, 52] # 24. black
]) / 255.0

class FinalProjectProcessor:
    def __init__(self):
        # [Source: 23] RAW檔格式
        self.height = 4180
        self.width = 6264
        self.ccm = np.eye(3)
        self.wb_gains = (1.0, 1.0, 1.0)
        self.calibration_gain = 1.0
        self.raw_max = 65535.0
        self.target_lab = self.precompute_target_lab()
    
    def precompute_target_lab(self):
        mask = TARGET_SRGB <= 0.04045
        linear_srgb = np.zeros_like(TARGET_SRGB)
        linear_srgb[mask] = TARGET_SRGB[mask] / 12.92
        linear_srgb[~mask] = ((TARGET_SRGB[~mask] + 0.055) / 1.055) ** 2.4
        return self.rgb_to_lab(linear_srgb.reshape(1, 24, 3)).reshape(24, 3)
    
    def loadraw(self, filepath):
        if not os.path.exists(filepath):
            print(f"Error: File not found {filepath}")
            return None
        data = np.fromfile(filepath, dtype=np.uint16)
        if data.size != self.height * self.width:
            h = int(np.sqrt(data.size * (self.height/self.width)))
            w = int(data.size / h)
            data = data.reshape(h, w)
        else:
            data = data.reshape(self.height, self.width)
        return data
    
    def demosaic(self, raw):
        """Step 1: Demosaic 與 自動黑電平扣除"""
        # 1. 自動估算黑電平
        estimated_black_level = np.percentile(raw, 0.5)
        # 2. 扣除黑電平
        raw = raw.astype(np.float32) - estimated_black_level
        raw = np.maximum(raw, 0)
        # 3. 更新最大值
        current_max = np.max(raw)
        if current_max > 0:
            self.raw_max = float(current_max)
        else:
            self.raw_max = 65535.0
        # 4. 正規化
        raw_norm = raw / self.raw_max
        raw16u = (raw_norm * 65535).astype(np.uint16)
        # 使用 BG2BGR
        demosaic = cv2.cvtColor(raw16u, cv2.COLOR_BayerBG2BGR)
        return demosaic.astype(np.float32) / 65535.0
    
    def n_white_balancing(self, img, block_size=3):
        """
        [論文實現] N-White Balancing for Multiple Illuminants
        參考: IEEE Access 2022, "N-White Balancing: White Balancing for Multiple 
              Illuminants Including Non-Uniform Illumination"
        """
        h, w = img.shape[:2]
        block_h = h // block_size
        block_w = w // block_size
        
        # Step 1: Block-wise illuminant estimation (使用 Grey-World)
        source_white_points = []
        coordinates = []
        
        print(f"  [N-White Balancing] Estimating {block_size}x{block_size} = {block_size**2} source white points...")
        
        for i in range(block_size):
            for j in range(block_size):
                # 計算 block 範圍
                y_start = i * block_h
                y_end = (i + 1) * block_h if i < block_size - 1 else h
                x_start = j * block_w
                x_end = (j + 1) * block_w if j < block_size - 1 else w
                
                block = img[y_start:y_end, x_start:x_end]
                
                # Grey-World 估計該 block 的 illuminant
                mean_b = np.mean(block[:, :, 0])
                mean_g = np.mean(block[:, :, 1])
                mean_r = np.mean(block[:, :, 2])
                
                # Source white point (BGR format)
                source_white_points.append([mean_b, mean_g, mean_r])
                
                # Block 中心座標
                center_y = (y_start + y_end) // 2
                center_x = (x_start + x_end) // 2
                coordinates.append([center_x, center_y])
        
        source_white_points = np.array(source_white_points)
        coordinates = np.array(coordinates)
        N = len(source_white_points)
        
        # === 印出 9 個 source white points 的 RGB 值 ===
        print(f"\n  [Source White Points] BGR values for {N} points:")
        for idx in range(N):
            b, g, r = source_white_points[idx]
            print(f"    Point {idx+1}: R={r:.4f}, G={g:.4f}, B={b:.4f}")
        print()
        
        print(f"  [N-White Balancing] Applying spatial weighted correction...")
        
        # Step 2: 為每個像素計算距離加權
        # 建立像素座標網格
        x_coords = np.arange(w)
        y_coords = np.arange(h)
        xx, yy = np.meshgrid(x_coords, y_coords)
        pixel_coords = np.stack([xx, yy], axis=-1)  # (h, w, 2)
        
        # 計算每個像素到所有 source white points 的距離
        distances = np.zeros((h, w, N))
        for m in range(N):
            diff = pixel_coords - coordinates[m]  # (h, w, 2)
            distances[:, :, m] = np.sqrt(np.sum(diff**2, axis=-1)) + 1e-6  # 避免除零
        
        # 計算權重 (距離倒數的歸一化)
        inv_distances = 1.0 / distances  # (h, w, N)
        weights = inv_distances / np.sum(inv_distances, axis=-1, keepdims=True)  # (h, w, N)
        
        # Step 3: 計算加權後的白平衡增益
        # 對每個 source white point 計算 gain
        gains = np.zeros((N, 3))
        for m in range(N):
            mean_g = source_white_points[m, 1]
            gains[m, 0] = mean_g / (source_white_points[m, 2] + 1e-6)  # R gain
            gains[m, 1] = 1.0  # G gain
            gains[m, 2] = mean_g / (source_white_points[m, 0] + 1e-6)  # B gain
        
        # Step 4: 對每個像素應用加權校正
        img_wb = img.copy()
        for c in range(3):  # BGR channels
            # weights: (h, w, N), gains: (N, 3)
            weighted_gain = np.sum(weights * gains[:, c], axis=-1)  # (h, w)
            img_wb[:, :, c] *= weighted_gain
        
        # 計算平均 gain 用於顯示
        avg_gain = np.mean(gains, axis=0)
        print(f"  [N-White Balancing] Average Gains: R={avg_gain[0]:.4f}, G={avg_gain[1]:.4f}, B={avg_gain[2]:.4f}")
        
        return np.clip(img_wb, 0, 1)
    
    def analyze_gray_card(self, gray_raw_path):
        """從 GrayCard.raw 解馬賽克並計算白平衡增益"""
        print(f"\n=== Step 1: 分析 GrayCard ===")
        raw = self.loadraw(gray_raw_path)
        img = self.demosaic(raw)
        
        h, w, _ = img.shape
        center_h, center_w = h // 2, w // 2
        crop_size = 500
        center_region = img[center_h-crop_size:center_h+crop_size, center_w-crop_size:center_w+crop_size]
        
        mean_b = np.mean(center_region[:, :, 0])
        mean_g = np.mean(center_region[:, :, 1])
        mean_r = np.mean(center_region[:, :, 2])
        
        r_gain = mean_g / (mean_r + 1e-6)
        b_gain = mean_g / (mean_b + 1e-6)
        g_gain = 1.0
        
        self.wb_gains = (r_gain, g_gain, b_gain)
        print(f"白平衡增益 (Gray Card): R={r_gain:.4f}, G={g_gain:.4f}, B={b_gain:.4f}")
    
    def apply_white_balance(self, img, custom_gains=None):
        """套用白平衡"""
        img_wb = img.copy()
        gains = custom_gains if custom_gains is not None else self.wb_gains
        img_wb[:, :, 0] *= gains[2]  # B
        img_wb[:, :, 1] *= gains[1]  # G
        img_wb[:, :, 2] *= gains[0]  # R
        return np.clip(img_wb, 0, 1)
    
    def rgb_to_lab(self, img_rgb):
        M = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        shape = img_rgb.shape
        reshaped_rgb = img_rgb.reshape(-1, 3)
        XYZ = np.dot(reshaped_rgb, M.T)
        
        Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
        X, Y, Z = XYZ[:, 0] / Xn, XYZ[:, 1] / Yn, XYZ[:, 2] / Zn
        
        def f_func(t): return np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16/116)
        
        L = 116 * f_func(Y) - 16
        a = 500 * (f_func(X) - f_func(Y))
        b = 200 * (f_func(Y) - f_func(Z))
        
        return np.stack([L, a, b], axis=1).reshape(shape)
    
    def extract_color_patches(self, img):
        patches_rgb = []
        start_x = 1300
        start_y = 930
        block_w = 400
        block_h = 400
        gap_x = 240
        gap_y = 240
        cols, rows = 6, 4
        
        debug_img = img.copy()
        for r in range(rows):
            for c in range(cols):
                x = start_x + c * (block_w + gap_x)
                y = start_y + r * (block_h + gap_y)
                patch = img[y:y+block_h, x:x+block_w, :]
                mean_bgr = np.mean(patch, axis=(0, 1))
                patches_rgb.append(mean_bgr[::-1])
                cv2.rectangle(debug_img, (x, y), (x+block_w, y+block_h), (0, 0, 1), 10)
        
        cv2.imwrite("processed_output12/debug_patches.png", np.clip(debug_img*255, 0, 255).astype(np.uint8))
        return np.array(patches_rgb)
    
    def objective_function(self, params, source_rgb, target_lab):
        ccm = params.reshape(3, 3)
        corrected = np.dot(source_rgb, ccm.T)
        corrected = np.clip(corrected, 0, 1)
        calculated_lab = self.rgb_to_lab(corrected.reshape(1, -1, 3)).reshape(-1, 3)
        delta = calculated_lab - target_lab
        return np.mean(np.sqrt(np.sum(delta**2, axis=1)))
    
    def row_sum_constraint(self, params):
        """等式約束：每一列的和必須等於 1"""
        ccm = params.reshape(3, 3)
        row_sums = np.sum(ccm, axis=1)  # 計算每列的和
        return row_sums - 1.0  # 返回偏差（應為 0）
    
    def optimize_ccm(self, color_checker_wb):
        """計算 CCM (使用白平衡後的 ColorChecker)，確保每列和為 1"""
        print("\n=== Step 3: 計算 CCM (約束：每列和為1) ===")
        
        source_rgb = self.extract_color_patches(color_checker_wb)
        
        if len(source_rgb) != 24:
            print("Error: 未能提取 24 個色塊。")
            return
        
        target_n5_g = 0.20
        source_n5_g = source_rgb[21][1]
        self.calibration_gain = target_n5_g / (source_n5_g + 1e-6)
        print(f"ColorChecker Calibration Gain: {self.calibration_gain:.4f}")
        
        source_rgb_norm = source_rgb * self.calibration_gain
        print("已正規化亮度。開始最佳化 CCM (使用 SLSQP 方法)...")
        
        initial_guess = np.eye(3).flatten()
        
        # 定義等式約束
        constraint = {
            'type': 'eq',
            'fun': self.row_sum_constraint
        }
        
        result = minimize(
            self.objective_function,
            initial_guess,
            args=(source_rgb_norm, self.target_lab),
            method='SLSQP',  # 改用 SLSQP，支持等式約束
            bounds=[(-3, 3)] * 9,
            constraints=constraint
        )
        
        self.ccm = result.x.reshape(3, 3)
        print(f"最佳化完成。Final Loss: {result.fun:.4f}")
        print("\nCCM 矩陣:")
        print(self.ccm)
        
        # 驗證每列的和
        print("\n每列和驗證:")
        for i in range(3):
            row_sum = np.sum(self.ccm[i, :])
            print(f"  Row {i+1}: {self.ccm[i, 0]:.4f} + {self.ccm[i, 1]:.4f} + {self.ccm[i, 2]:.4f} = {row_sum:.6f}")
    
    def apply_ccm(self, img, manual_gain=None):
        shape = img.shape
        img_flat = img.reshape(-1, 3)
        img_rgb = img_flat[:, [2, 1, 0]]  # BGR to RGB
        
        if manual_gain is not None:
            gain = manual_gain
            print(f"  [CCM Step] Applying Manual Gain: {gain:.4f}")
        else:
            current_mean = np.mean(img_rgb[:, 1])
            target_mean = 0.18
            if current_mean < 1e-6: gain = 1.0
            else: gain = target_mean / current_mean
            gain = min(gain, 30.0)
            print(f"  [CCM Step] Adaptive Exposure Gain: {gain:.4f} (Mean: {current_mean:.5f})")
        
        img_rgb = img_rgb * gain
        corrected = np.dot(img_rgb, self.ccm.T)
        corrected = np.clip(corrected, 0, 1)
        
        return corrected[:, [2, 1, 0]].reshape(shape)
    
    def tonereproduction(self, img):
        img = img.astype(np.float32)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        Lw = img_gray + 1e-6
        log_avg = np.exp(np.mean(np.log(Lw)))
        key = 0.18
        L = (key / log_avg) * Lw
        Lwhite = np.percentile(L, 99.5)
        Ld = (L * (1 + L / (Lwhite**2))) / (1 + L)
        scale = Ld / Lw
        return np.clip(img * scale[:, :, np.newaxis], 0, 1)
    
    def color_enhancement(self, img, factor=1.2):
        """色彩增強：轉換至 CIELab 空間，拉伸 a* 與 b* 通道"""
        img_u8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2Lab).astype(np.float32)
        
        lab[:, :, 1] = (lab[:, :, 1] - 128) * factor + 128
        lab[:, :, 2] = (lab[:, :, 2] - 128) * factor + 128
        
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR).astype(np.float32) / 255.0
        return img_bgr
    
    def gamma_correction(self, img, gamma=1/2.2):
        return np.clip(np.power(img, gamma), 0, 1)
    
    def save_image(self, img, filename):
        save_path = os.path.join("processed_output12", filename)
        cv2.imwrite(save_path, np.clip(img * 255, 0, 255).astype(np.uint8))
        print(f"Saved: {save_path}")
    
    def run_pipeline(self, raw_file, base_name, manual_gain=None):
        """完整 Pipeline for IMG_000x"""
        print(f"\n=== Processing {raw_file} ===")
        raw = self.loadraw(raw_file)
        if raw is None: return
        
        # 1. Demosaic
        img = self.demosaic(raw)
        raw_vis = raw.astype(np.float32) / 65535.0
        self.save_image(raw_vis, f"{base_name}_step1_Raw.jpg")
        self.save_image(img, f"{base_name}_step2_Demosaic.jpg")
        
        # 2. N-White Balance (論文方法)
        img_wb = self.n_white_balancing(img, block_size=3)
        self.save_image(img_wb, f"{base_name}_step3_NWhiteBalance.jpg")
        
        # 3. CCM + Exposure
        img_cc = self.apply_ccm(img_wb, manual_gain=manual_gain)
        self.save_image(img_cc, f"{base_name}_step4_CCM.jpg")
        
        # 4. Tone Reproduction
        img_tr = self.tonereproduction(img_cc)
        self.save_image(img_tr, f"{base_name}_step5_ToneReproduction.jpg")
        
        # 5. Enhancement
        img_en = self.color_enhancement(img_tr)
        self.save_image(img_en, f"{base_name}_step6_Enhancement.jpg")
        
        # 6. Final
        img_final = self.gamma_correction(img_en)
        self.save_image(img_final, f"{base_name}_final.jpg")


if __name__ == "__main__":
    os.makedirs("processed_output12", exist_ok=True)
    processor = FinalProjectProcessor()
    
    # Step 1: 解馬賽克 ColorChecker 和 GrayCard
    colorchecker_demosaiced = None
    if os.path.exists("ColorChecker.raw"):
        print("\n=== Step 0: 解馬賽克 ColorChecker ===")
        raw = processor.loadraw("ColorChecker.raw")
        colorchecker_demosaiced = processor.demosaic(raw)
        print("ColorChecker 解馬賽克完成")
    
    if os.path.exists("GrayCard.raw"):
        # 計算白平衡增益
        processor.analyze_gray_card("GrayCard.raw")
    
    # Step 2: 套用白平衡到 ColorChecker 並輸出對比圖
    if colorchecker_demosaiced is not None:
        print("\n=== Step 2: 套用白平衡到 ColorChecker ===")
        # Before (解馬賽克)
        processor.save_image(processor.gamma_correction(colorchecker_demosaiced),
                           "ColorChecker_WB_Before.jpg")
        
        # After (解馬賽克 + 白平衡)
        colorchecker_wb = processor.apply_white_balance(colorchecker_demosaiced)
        processor.save_image(processor.gamma_correction(colorchecker_wb),
                           "ColorChecker_WB_After.jpg")
        
        # Step 3: 用白平衡後的 ColorChecker 計算 CCM
        processor.optimize_ccm(colorchecker_wb)
        
        # Step 4: 輸出 CCM 校正對比圖
        print("\n=== Step 4: 輸出 ColorChecker CCM 校正對比圖 ===")
        # Before (白平衡後)
        processor.save_image(processor.gamma_correction(colorchecker_wb),
                           "ColorChecker_CCM_Before.jpg")
        
        # After (白平衡 + CCM)
        colorchecker_ccm = processor.apply_ccm(colorchecker_wb,
                                              manual_gain=processor.calibration_gain)
        processor.save_image(processor.gamma_correction(colorchecker_ccm),
                           "ColorChecker_CCM_After.jpg")
    
    print("\n--- 開始處理生活照 (使用 N-White Balancing) ---")
    
    # Step 5: 生活照處理 (IMG_0001~7) - 使用論文的 N-White Balancing
    for i in range(1, 8):
        f = f"IMG_{i:04d}.raw"
        if os.path.exists(f):
            processor.run_pipeline(f, f.split('.')[0], manual_gain=None)
