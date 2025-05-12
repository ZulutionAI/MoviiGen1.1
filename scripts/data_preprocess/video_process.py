import csv
import glob
import json
import os
import subprocess
import sys
from typing import Any, Dict, List

import cv2


def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """使用OpenCV获取视频元数据"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频文件")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        'width': width,
        'height': height,
        'fps': fps,
        'duration': duration
    }

def process_csv(csv_path: str) -> Dict[str, str]:
    """处理CSV文件并创建文件名到描述的映射"""
    filename_to_desc = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        required_columns = {'Filename', 'Video Description'}
        if not required_columns.issubset(reader.fieldnames):
            missing = required_columns - set(reader.fieldnames)
            raise ValueError(f"CSV缺少必要列：{missing}")
        
        for row in reader:
            filename = row['Filename'].strip()
            desc = row['Video Description'].strip()
            if filename in filename_to_desc:
                print(f"警告：重复文件名 {filename} 在 {csv_path}")
            filename_to_desc[filename] = desc
    return filename_to_desc

def main(src_dir: str, output_path: str = 'output.json'):
    result = []
    
    for root, _, files in os.walk(src_dir):
        # 查找CSV文件
        csv_files = glob.glob(os.path.join(root, '*.csv'))
        if not csv_files:
            continue
            
        csv_path = csv_files[0]  # 取第一个CSV文件
        if len(csv_files) > 1:
            print(f"警告：多个CSV文件，使用 {csv_path}")
        
        try:
            filename_map = process_csv(csv_path)
        except Exception as e:
            print(f"CSV处理失败 {csv_path}: {str(e)}")
            continue
        
        # 处理视频文件
        for video_path in glob.glob(os.path.join(root, '*.mp4')):
            filename = os.path.basename(video_path)
            desc = filename_map.get(filename, "")
            
            try:
                metadata = get_video_metadata(video_path)
            except Exception as e:
                print(f"视频处理失败 {video_path}: {str(e)}")
                continue
            
            # 生成相对路径
            relative_path = os.path.relpath(video_path, src_dir).replace('\\', '/')
            
            result.append({
                "path": relative_path,
                "resolution": {
                    "width": metadata['width'],
                    "height": metadata['height']
                },
                "fps": metadata['fps'],
                "duration": metadata['duration'],
                "cap": [desc] if desc else []
            })
    
    # 写入JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("使用方法:python script.py <src_dir> [output.json]")
        sys.exit(1)
    
    src_directory = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'output.json'
    main(src_directory, output_file)