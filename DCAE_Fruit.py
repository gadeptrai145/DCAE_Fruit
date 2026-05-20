import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

#CẤU HÌNH CÁC THAM SỐ CHÍNH
IMG_SIZE = 256
CHANNELS = 3
LATENT_DIM = 64
EPOCHS = 30
BATCH_SIZE = 32
SEED = 42
EXCEL_LOG_PATH = "training_performance_logs.xlsx"

#CALLBACK
callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
            ),

        tf.keras.callbacks.ModelCheckpoint(
            "best_autoencoder.h5",
            monitor='val_loss',
            save_best_only=True
            ),

        tf.keras.callbacks.ModelCheckpoint(
            "best_autoencoder.keras",
            monitor='val_loss',
            save_best_only=True),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3
            )
    ]

#KIẾN TRÚC MẠNG CONVOLUTIONAL AUTOENCODER
def build_convolutional_autoencoder():
    input_img = layers.Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS))
    
    # Encoder (Trích xuất đặc trưng & Nén)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Dropout(0.35)(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Dropout(0.2)(x)
    
    encoded = layers.Conv2D(LATENT_DIM, (3, 3), activation='relu', padding='same')(x)
    
    # Decoder (Giải mã & Khôi phục bề mặt)
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(encoded)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
    decoded = layers.Conv2D(CHANNELS, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

#HÀM CHUẨN HÓA CẤU TRÚC ĐẦU VÀO/ĐẦU RA CHO AUTOENCODER
def preprocess_for_autoencoder(images):
    # Chuẩn hóa pixel về [0, 1]
    images = tf.cast(images, tf.float32) / 255.0
    noise = tf.random.normal(
        shape=tf.shape(images),
        mean=0,
        stddev=0.1)

    noisy=images+noise
    noisy = tf.clip_by_value(noisy,0,1)
    return noisy, images  # Trả về cặp (X, X) vì Autoencoder học tự tái tạo

#MAIN
if __name__ == "__main__":
    dataset_good_path = r"Fruit_data\Good Quality_Fruits" 
    
    if not os.path.exists(dataset_good_path):
        print(f"[LỖI] Không tìm thấy thư mục dữ liệu tại: {dataset_good_path}")
        exit()

    print("[BƯỚC 1] Đang khởi tạo luồng đọc dữ liệu từ thư mục (Chống tràn RAM)...")
    
    #TẢI TẬP TRAIN (80% DỮ LIỆU)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_good_path,
        shuffle=True,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode=None  # Không lấy nhãn phân loại lớp
    )

    #Tải tập Validation (20% dữ liệu)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_good_path,
        shuffle=True,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode=None
    )

    print("\n[BƯỚC 2] Đang kích hoạt ánh xạ chuẩn hóa hình ảnh [0, 1] theo luồng...")
    train_ds = train_ds.map(
        preprocess_for_autoencoder,
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.map(
        preprocess_for_autoencoder,
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    print("\n[BƯỚC 3] Khởi tạo kiến trúc mạng Convolutional Autoencoder...")
    model = build_convolutional_autoencoder()
    
    # Hiển thị cấu trúc mô hình chi tiết
    print("\n" + "="*40 + " KIẾN TRÚC MÔ HÌNH (SUMMARY) " + "="*40)
    model.summary()
    print("="*109 + "\n")
    
    print("[BƯỚC 4] Bắt đầu quá trình huấn luyện mạng nơ-ron...")
    print("-" * 80)
    
    # Thực hiện huấn luyện bằng hệ thống Dataset luồng mới
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    hist_df = pd.DataFrame(history.history)
    hist_df.to_excel(EXCEL_LOG_PATH, index=False)
    print("-" * 80)
    
    #ĐÓNG GÓI MODEL VÀ TÍNH NGƯỠNG BẤT THƯỜNG
    print("\n[BƯỚC 5] Huấn luyện hoàn tất. Đang tính toán Ngưỡng chặn bất thường (Threshold)...")
    
    #ĐỂ TÍNH NGƯỠNG CHUẨN XÁC, TA CẦN QUÉT QUA TẬP TRAIN CŨ ĐỂ LẤY PHÂN VỊ LỖI
    print("-> Đang tính toán phân phối sai số tái tạo...")

    clean_images_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_good_path,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        shuffle=False,
        image_size=(IMG_SIZE,IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode=None
    )

    clean_images_ds = clean_images_ds.map(
        lambda x: tf.cast(x, tf.float32)/255.0
    )
    all_mses = []
    
    #QUÉT TỪNG BATCH CỦA TẬP DỮ LIỆU ĐỂ TÍNH MSE (TRÁNH TẢI TOÀN BỘ MẢNG LÊN RAM)
    for clean_batch in clean_images_ds:

        reconstructed=model(
            clean_batch,
            training=False
        )

        mse=tf.reduce_mean(
            tf.square(
                clean_batch-reconstructed
            ),
            axis=(1,2,3)
        )

        all_mses.extend(
            mse.numpy()
        )
        
    calculated_threshold = float(np.percentile(all_mses, 95))
    
    #XUẤT CẤU HÌNH VÀ MÔ HÌNH
    runtime_config = {
        "img_size": IMG_SIZE,
        "threshold": calculated_threshold
    }
    with open("model_config.json", "w") as config_file:
        json.dump(runtime_config, config_file)
        
    model.save("fruit_autoencoder_model.h5")
    model.save("fruit_autoencoder_model.keras")
    print("\n" + "="*60)
    print(" XUẤT CÁC FILE THÀNH CÔNG (KHÔNG TRÀN RAM):")
    print(f" 1. Nhật ký đồ thị (Excel): {EXCEL_LOG_PATH}")
    print(f" 2. Trọng số Model AI (.h5): fruit_autoencoder_model.h5")
    print(f" 3. Tham số Ngưỡng chặn (Ngưỡng: {calculated_threshold:.6f}): model_config.json")
    print("="*60)
