import argparse
import shutil
from pathlib import Path


def copy_csvs_preserve_structure(src_dir, dst_dir):
    """复制目录下所有子文件夹中的CSV文件，保持原始目录结构"""
    
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    
    # 递归查找所有CSV文件
    csv_files = list(src_path.rglob("*.csv"))
    
    if not csv_files:
        print(f"未找到CSV文件: {src_dir}")
        return

    print(f"共发现 {len(csv_files)} 个CSV文件")

    for csv_file in csv_files:
        # 计算相对路径
        relative_path = csv_file.relative_to(src_path)
        
        # 构建目标路径
        target_file = dst_path / relative_path
        
        # 创建目标目录（自动处理多级嵌套）
        target_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 执行文件复制
        shutil.copy2(csv_file, target_file)
        print(f"已复制: {csv_file} -> {target_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy CSV files and preserve directory structure")
    parser.add_argument("--src_dir", help="Source directory path",default=None)
    parser.add_argument("--dst_dir", help="Destination directory path",default=None)
    args = parser.parse_args()
    copy_csvs_preserve_structure(args.src_dir, args.dst_dir)
