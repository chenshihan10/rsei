import os
from tqdm import tqdm
from core.rsei_core import compute_rsei_for_year, export_all_outputs

DATA_DIR = 'data'
OUTPUT_DIR = 'output'

def get_year_dirs(data_dir):
    """获取 data/ 目录下所有年份文件夹路径"""
    return sorted([
        os.path.join(data_dir, d) for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

def main():
    year_dirs = get_year_dirs(DATA_DIR)

    print("🛰 开始处理年份数据：")
    for year_dir in tqdm(year_dirs, desc="RSEI计算进度", unit="年"):
        year = os.path.basename(year_dir)
        try:
            rsei, profile, factors = compute_rsei_for_year(year_dir)
            export_dir = os.path.join(OUTPUT_DIR, year)
            export_all_outputs(rsei, factors, profile, export_dir)
        except Exception as e:
            print(f"❌ {year} 处理失败: {e}")
        else:
            print(f"✅ {year} 完成，结果已保存至 {export_dir}")

if __name__ == '__main__':
    main()
