"""
fix_and_zip.py
==============
1. Sửa tất cả lỗi format trong thư mục submission/
2. Báo cáo file nào thiếu
3. Đóng gói submission.zip chuẩn format BTC
"""
import os, zipfile, re

SUB_DIR = "/home/khoaha/PyCharmMiscProject/khoa/aic_retrieval/submission"
ZIP_OUT = "/home/khoaha/PyCharmMiscProject/khoa/aic_retrieval/submission.zip"

# Tất cả query cần có
ALL_QUERIES = [
    "query-p1-1-kis.csv",  "query-p1-2-kis.csv",  "query-p1-4-trake.csv",
    "query-p1-5-kis.csv",  "query-p1-6-kis.csv",  "query-p1-7-kis.csv",
    "query-p1-8-kis.csv",  "query-p1-9-kis.csv",  "query-p1-10-kis.csv",
    "query-p1-11-kis.csv", "query-p1-12-kis.csv", "query-p1-13-kis.csv",
    "query-p1-14-kis.csv", "query-p1-15-qa.csv",  "query-p1-16-trake.csv",
    "query-p1-17-kis.csv", "query-p1-18-trake.csv","query-p1-19-qa.csv",
    "query-p1-20-kis.csv", "query-p1-21-kis.csv", "query-p1-22-qa.csv",
    "query-p1-23-kis.csv", "query-p1-24-kis.csv", "query-p1-25-kis.csv",
]

print("=" * 55)
print("🔧 Fix & Zip Submission")
print("=" * 55)

fixed_count = 0
skipped_count = 0

csv_files = [f for f in os.listdir(SUB_DIR) if f.endswith(".csv") and not f.startswith(".~")]

for fname in sorted(csv_files):
    fpath = os.path.join(SUB_DIR, fname)
    
    with open(fpath, "r", encoding="utf-8") as f:
        raw = f.read()

    # --- Fix 1: Chuẩn hóa line endings → \n ---
    fixed = raw.replace("\r\r\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
    
    lines = [l.rstrip() for l in fixed.split("\n")]
    
    new_lines = []
    changed = False
    skip_file = False
    
    for line in lines:
        if not line.strip():
            continue
        
        parts = line.split(",")
        
        # --- Fix 2: Xóa dấu phẩy thừa cuối dòng ---
        # Với KIS: chỉ cần 2 cột (video_id, frame_idx)
        # Với TRAKE: >=3 cột (video_id, frame1, frame2, ...)
        # Với Q&A: 3 cột (video_id, frame_idx, answer)
        
        if "-kis" in fname:
            # KIS: giữ đúng 2 cột
            if len(parts) >= 2:
                vid = parts[0].strip()
                fidx = parts[1].strip()
                
                # Bỏ dòng có video_id rỗng hoặc không hợp lệ
                if not vid or not re.match(r'^L\d+_V\d+$', vid):
                    print(f"   ⚠️  Bỏ dòng invalid video_id: '{line[:40]}'")
                    changed = True
                    continue
                
                # Bỏ frame_idx rỗng hoặc không phải số
                if not fidx or not fidx.isdigit():
                    print(f"   ⚠️  Bỏ dòng invalid frame_idx: '{line[:40]}'")
                    changed = True
                    continue
                    
                clean = f"{vid},{fidx}"
                if clean != line:
                    changed = True
                new_lines.append(clean)
            else:
                print(f"   ⚠️  Bỏ dòng quá ngắn: '{line[:40]}'")
                changed = True
                
        elif "-qa" in fname:
            # Q&A: giữ đúng 3 cột
            if len(parts) >= 2:
                vid = parts[0].strip()
                fidx = parts[1].strip()
                answer = ",".join(parts[2:]).strip() if len(parts) > 2 else ""
                
                if not vid or not re.match(r'^L\d+_V\d+$', vid):
                    changed = True; continue
                if not fidx or not fidx.isdigit():
                    changed = True; continue
                    
                if answer and ("," in answer or '"' in answer):
                    answer = '"' + answer.strip('"').replace('"', '""') + '"'
                    
                clean = f"{vid},{fidx},{answer}"
                if clean != line:
                    changed = True
                new_lines.append(clean)
                
        elif "-trake" in fname:
            # TRAKE: giữ nguyên (format đã đúng \n)
            if len(parts) >= 2:
                vid = parts[0].strip()
                if not vid or not re.match(r'^L\d+_V\d+$', vid):
                    changed = True; continue
                clean = ",".join(p.strip() for p in parts)
                if clean != line:
                    changed = True
                new_lines.append(clean)
        else:
            new_lines.append(line)
    
    result = "\n".join(new_lines) + "\n"
    
    if changed or result != raw:
        with open(fpath, "w", encoding="utf-8", newline="") as f:
            f.write(result)
        print(f"✅ Fixed: {fname} ({len(new_lines)} dòng)")
        fixed_count += 1
    else:
        print(f"✓  OK   : {fname} ({len(new_lines)} dòng)")

# Báo cáo file thiếu
print()
print("─" * 55)
existing = set(f for f in os.listdir(SUB_DIR) if f.endswith(".csv") and not f.startswith(".~"))
missing = [q for q in ALL_QUERIES if q not in existing]
if missing:
    print(f"⚠️  THIẾU {len(missing)} file (chưa chạy query):")
    for m in missing:
        print(f"   ✗ {m}")
else:
    print("✅ Đủ tất cả 24 file CSV!")

# Tạo ZIP
print()
print("─" * 55)
print("📦 Đang tạo submission.zip...")
with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in sorted(existing):
        fpath = os.path.join(SUB_DIR, fname)
        arcname = f"submission/{fname}"
        zf.write(fpath, arcname)
        
zip_size = os.path.getsize(ZIP_OUT)
print(f"✅ Tạo xong: submission.zip ({zip_size/1024:.1f} KB)")
print()
print("Nội dung zip:")
with zipfile.ZipFile(ZIP_OUT, "r") as zf:
    for name in sorted(zf.namelist()):
        info = zf.getinfo(name)
        print(f"   {name} ({info.file_size} bytes)")

print()
print("=" * 55)
print(f"✅ Fixed {fixed_count} file(s)")
if missing:
    print(f"⚠️  Thiếu {len(missing)} file — xem danh sách ở trên")
print(f"📦 File nộp: submission.zip")
print("=" * 55)
