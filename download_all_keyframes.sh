#!/bin/bash
cd /home/khoaha/PyCharmMiscProject/khoa/aic_retrieval/data/keyframes

download_and_extract() {
    URL=$1
    ZIP_NAME=$(basename $URL)
    echo "========================================"
    echo "Bắt đầu tải $ZIP_NAME..."
    if wget -q --show-progress -U "Mozilla/5.0" -c "$URL" -O "$ZIP_NAME"; then
        echo "Đang giải nén $ZIP_NAME..."
        unzip -q "$ZIP_NAME" -d .
        
        # Di chuyển ra ngoài nếu bị bọc trong thư mục 'keyframes'
        if [ -d "keyframes" ]; then
            mv keyframes/* . 2>/dev/null
            rm -rf keyframes
        fi
        
        # Xoá zip để giải phóng ổ cứng
        rm -f "$ZIP_NAME"
        echo "✅ Hoàn tất $ZIP_NAME!"
    else
        echo "❌ Lỗi khi tải $ZIP_NAME (File không tồn tại hoặc lỗi mạng)"
        rm -f "$ZIP_NAME"
    fi
}

echo "=== TIẾN TRÌNH TẢI TỰ ĐỘNG KEYFRAMES (25GB) ==="

# Tải các gói lẻ
for i in 22 24 25 27 28 29; do
    download_and_extract "https://aic-data.ledo.io.vn/Keyframes_L${i}.zip"
done

# Xử lý riêng gói L26 (thường bị chia nhỏ)
echo "Đang kiểm tra gói L26..."
if wget -q --spider -U "Mozilla/5.0" "https://aic-data.ledo.io.vn/Keyframes_L26.zip"; then
    download_and_extract "https://aic-data.ledo.io.vn/Keyframes_L26.zip"
else
    for p in a b c d e; do
        download_and_extract "https://aic-data.ledo.io.vn/Keyframes_L26_${p}.zip"
    done
fi

echo "🎉 TẤT CẢ HOÀN TẤT!"
