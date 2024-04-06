import cv2
import numpy as np
from pathlib import Path

def compare_fingerprints(uploaded_img_path, database_folder_path):
    uploaded_img = cv2.imread(uploaded_img_path, cv2.IMREAD_GRAYSCALE)
    
    if uploaded_img is None:
        print("上傳的指紋圖像無法加載。")
        return
    
    # 獲取數據庫中所有的指紋圖像路徑
    fingerprint_paths = list(Path(database_folder_path).glob('*.jpg'))
    
    for fingerprint_path in fingerprint_paths:
        database_img = cv2.imread(str(fingerprint_path), cv2.IMREAD_GRAYSCALE)
        
        if database_img is None:
            print(f"數據庫中的指紋圖像 {fingerprint_path} 無法加載。")
            continue
        
        # 確保兩個圖像大小相同
        database_img = cv2.resize(database_img, (uploaded_img.shape[1], uploaded_img.shape[0]))
        
        # 計算兩個圖像之間的相似度
        similarity = cv2.matchTemplate(uploaded_img, database_img, cv2.TM_CCOEFF_NORMED)
        max_similarity = np.max(similarity)
        
        if max_similarity >= 0.8:
            print(f"比對成功：{fingerprint_path}")
            return  # 找到匹配，結束函數
    
    print("沒有找到匹配的指紋。")

# 使用示例
compare_fingerprints('images.jpg', 'database_fingerprints')