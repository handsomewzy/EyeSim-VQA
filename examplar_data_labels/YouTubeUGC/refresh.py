# import os

# # 视频目录
# video_dir = '/data1/userhome/luwen/Code/wzy/VQA_dataset/KVQ/train_video-001'  # 修改为你的视频目录

# # 原始 label 文件
# input_file = '/data1/userhome/luwen/Code/wzy/DOVER-master/examplar_data_labels/KVQ/KVQ_train.txt'
# # 输出的过滤后 label 文件
# output_file = '/data1/userhome/luwen/Code/wzy/DOVER-master/examplar_data_labels/KVQ/KVQ_train_refresh.txt'

# # 读取并处理
# with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
#     for line in fin:
#         line = line.strip()
#         if not line:
#             continue
#         video_name = line.split(',')[0]
#         video_path = os.path.join(video_dir, video_name)
#         if os.path.isfile(video_path):
#             fout.write(line + '\n')

# print(f"过滤完成，已保存至 {output_file}")


import os
import subprocess
from multiprocessing import Pool, cpu_count

# 配置路径
video_dir = '/data1/userhome/luwen/Code/wzy/VQA_dataset/KVQ/train_video-001'
input_file = '/data1/userhome/luwen/Code/wzy/DOVER-master/examplar_data_labels/KVQ/KVQ_train.txt'
output_file = '/data1/userhome/luwen/Code/wzy/DOVER-master/examplar_data_labels/KVQ/KVQ_train_refresh.txt'
log_bad_file = '/data1/userhome/luwen/Code/wzy/DOVER-master/examplar_data_labels/KVQ/KVQ_train_bad.txt'

# ffprobe 检测函数（快速）
def is_video_readable_ffprobe(video_path):
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
        return result.returncode == 0 and result.stdout.strip() != b""
    except:
        return False

# 每一行的检测逻辑（用于并行）
def check_line_valid(line):
    line = line.strip()
    if not line:
        return None
    video_name = line.split(',')[0]
    video_path = os.path.join(video_dir, video_name)

    if os.path.isfile(video_path) and is_video_readable_ffprobe(video_path):
        return ("valid", line)
    else:
        return ("invalid", video_name)

# 主函数
def main():
    with open(input_file, 'r') as fin:
        lines = [line.strip() for line in fin if line.strip()]

    print(f"📦 开始检查 {len(lines)} 个视频路径...（并行处理）")
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(check_line_valid, lines)

    valid_lines = []
    bad_videos = []
    for result in results:
        if result is None:
            continue
        tag, content = result
        if tag == "valid":
            valid_lines.append(content)
        else:
            bad_videos.append(content)

    # 写入过滤后的 label 文件
    with open(output_file, 'w') as fout:
        for line in valid_lines:
            fout.write(line + '\n')

    # 写入坏视频日志
    with open(log_bad_file, 'w') as ferr:
        for vid in bad_videos:
            ferr.write(vid + '\n')

    print(f"\n✅ 完成过滤：共 {len(valid_lines)} 个有效视频，{len(bad_videos)} 个无效视频")
    print(f"✔ 结果保存至：{output_file}")
    print(f"⚠️ 无法打开的视频列表写入：{log_bad_file}")

if __name__ == "__main__":
    main()