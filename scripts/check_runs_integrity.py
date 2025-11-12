import os
import glob
import shutil

# --- Cấu hình ---
RUNS_DIR = 'data/runs'
ESSENTIAL_FILES = [
    'traffic_edges.json',
    'weather_snapshot.json',
    'nodes.json',
    'edges.json'
]
# Đặt thành True nếu bạn muốn script tự động xóa các thư mục lỗi
AUTO_DELETE_INVALID = True

def main():
    print(f"--- Bắt đầu kiểm tra tính toàn vẹn của dữ liệu trong '{RUNS_DIR}' ---")

    run_dirs = sorted(glob.glob(os.path.join(RUNS_DIR, 'run_*')))
    
    if not run_dirs:
        print("Không tìm thấy thư mục 'run_*' nào để kiểm tra.")
        return

    valid_dirs = []
    invalid_dirs = []

    for run_dir in run_dirs:
        is_valid = True
        missing_files = []
        for filename in ESSENTIAL_FILES:
            file_path = os.path.join(run_dir, filename)
            if not os.path.exists(file_path):
                is_valid = False
                missing_files.append(filename)
        
        if is_valid:
            valid_dirs.append(run_dir)
        else:
            invalid_dirs.append((run_dir, missing_files))

    # --- In ra Báo cáo ---
    print("\n--- Báo cáo kiểm tra ---")
    print(f"Tổng số thư mục 'run_*' tìm thấy: {len(run_dirs)}")
    print(f"Số thư mục hợp lệ (đủ file): {len(valid_dirs)}")
    print(f"Số thư mục KHÔNG hợp lệ (thiếu file): {len(invalid_dirs)}")

    if invalid_dirs:
        print("\nChi tiết các thư mục không hợp lệ:")
        for dir_path, files in invalid_dirs:
            print(f"- Thư mục: {dir_path}")
            print(f"  -> Thiếu file: {', '.join(files)}")
    
    # --- Tùy chọn Xóa ---
    if invalid_dirs and AUTO_DELETE_INVALID:
        print("\n--- Tự động xóa các thư mục không hợp lệ ---")
        confirm = input("Bạn có chắc chắn muốn xóa vĩnh viễn {} thư mục không hợp lệ không? (yes/no): ".format(len(invalid_dirs)))
        if confirm.lower() == 'yes':
            for dir_path, _ in invalid_dirs:
                try:
                    shutil.rmtree(dir_path)
                    print(f"Đã xóa: {dir_path}")
                except OSError as e:
                    print(f"Lỗi khi xóa {dir_path}: {e}")
            print("Hoàn tất dọn dẹp.")
        else:
            print("Đã hủy thao tác xóa.")
    elif invalid_dirs:
        print("\nĐể tự động xóa, hãy đặt AUTO_DELETE_INVALID = True trong script và chạy lại.")

if __name__ == '__main__':
    main()