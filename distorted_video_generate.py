import os
import cv2
import numpy as np
import random
from tqdm import tqdm

# -------- 各种失真方法 --------
def add_gaussian_noise(img):
    std = random.uniform(5, 30)
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy

def add_motion_blur(img):
    k = random.choice(range(5, 26, 2))
    kernel = np.zeros((k, k))
    kernel[int((k - 1) / 2), :] = np.ones(k)
    kernel /= k
    return cv2.filter2D(img, -1, kernel)

def add_jpeg_compression(img):
    quality = random.randint(10, 50)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(encimg, 1)

def add_salt_pepper(img):
    amount = random.uniform(0.01, 0.05)
    noisy = img.copy()
    num_salt = int(amount * img.size * 0.5)
    coords = [np.random.randint(0, i, num_salt) for i in img.shape[:2]]
    noisy[coords[0], coords[1]] = 255

    num_pepper = int(amount * img.size * 0.5)
    coords = [np.random.randint(0, i, num_pepper) for i in img.shape[:2]]
    noisy[coords[0], coords[1]] = 0
    return noisy

def add_mean_blur(img):
    k = random.choice(range(3, 16, 2))
    return cv2.blur(img, (k, k))

# -------- 主处理函数 --------
def process_videos(input_dir, output_root_dir):
    distortion_funcs = {
        "gaussian_noise": add_gaussian_noise,
        "motion_blur": add_motion_blur,
        "jpeg_compression": add_jpeg_compression,
        "salt_pepper": add_salt_pepper,
        "mean_blur": add_mean_blur,
    }

    os.makedirs(output_root_dir, exist_ok=True)
    for dname in distortion_funcs.keys():
        os.makedirs(os.path.join(output_root_dir, dname), exist_ok=True)

    for fname in tqdm(os.listdir(input_dir), desc="Processing Videos"):
        if not fname.lower().endswith(('.mp4', '.avi', '.mov')):
            continue

        input_path = os.path.join(input_dir, fname)
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频: {fname}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writers = {}
        for dname in distortion_funcs.keys():
            out_path = os.path.join(output_root_dir, dname, fname)
            writers[dname] = cv2.VideoWriter(
                out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
            )

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            for dname, func in distortion_funcs.items():
                distorted = func(frame)
                writers[dname].write(distorted)

        cap.release()
        for writer in writers.values():
            writer.release()

        print(f"✅ 已处理并保存：{fname}")

# -------- 示例调用 --------
input_folder = "/data1/userhome/luwen/Code/wzy/VQA_dataset/KoNViD_1k_videos"
output_folder = "/data1/userhome/luwen/Code/wzy/VQA_dataset/KoNViD_1k_video_distorted"
process_videos(input_folder, output_folder)
