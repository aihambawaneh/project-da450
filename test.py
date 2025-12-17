import cv2
import numpy as np
import tensorflow as tf
import tempfile
from gtts import gTTS
import pygame

# ------------ تحميل الموديل ------------
model = tf.keras.models.load_model("997.keras")

# ------------ إعداد المعطيات ------------
H, W, C = model.input_shape[1:]
label_cod = {
    0: "Pepper__bell___Bacterial_spot",
    1: "Pepper__bell___healthy",
    2: "Potato___Early_blight",
    3: "Potato___Late_blight",
    4: "Potato___healthy",
    5: "Tomato_Bacterial_spot",
    6: "Tomato_Early_blight",
    7: "Tomato_Late_blight",
    8: "Tomato_Leaf_Mold",
    9: "Tomato_Septoria_leaf_spot",
    10: "Tomato_Spider_mites_Two_spotted_spider_mite",
    11: "Tomato__Target_Spot",
    12: "Tomato__Tomato_YellowLeaf__Curl_Virus",
    13: "Tomato__Tomato_mosaic_virus",
    14: "Tomato_healthy"
}


# ------------ دوال مساعدة ------------
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"الصورة غير موجودة: {img_path}")
    img = cv2.resize(img, (W, H))
    if C == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, -1)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, 0), img


def speak_tts(text, lang="en", slow=False):
    """ينطق النص باستخدام gTTS + pygame. ينطق اسم الفئة فقط."""
    # gTTS يكتب إلى ملف مؤقت
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        gTTS(text=text, lang=lang, slow=slow).write_to_fp(tmp)
        mp3_path = tmp.name
    # تشغيل بالصوت عبر pygame
    pygame.mixer.init()
    pygame.mixer.music.load(mp3_path)
    pygame.mixer.music.play()
    clock = pygame.time.Clock()
    while pygame.mixer.music.get_busy():
        clock.tick(10)


# ------------ المسار إلى الصورة ------------
image_path = r"C:\Users\Asus\Desktop\project cnn\projects\Potato-Disease\Potato-Disease-Classification\training\PlantVillage\Pepper__bell___healthy\0a3f2927-4410-46a3-bfda-5f4769a5aaf8___JR_HL 8275.JPG" # غيّرها لمسارك

# ------------ التنبؤ ------------
x, _ = preprocess_image(image_path)
pred = model.predict(x, verbose=0)[0]
class_id = int(np.argmax(pred))
confidence = float(np.max(pred))
label = label_cod.get(class_id, "Unknown")

print(f"النتيجة: {label} (الثقة: {confidence:.2f})")

# ------------ عرض الصورة مع النتيجة ------------
vis = cv2.imread(image_path)
cv2.putText(
    vis,
    f"{label} ({confidence:.2f})",
    (10, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2,
)
cv2.imshow("Result", vis)

# ------------ النطق بالـ TTS ------------
# اللغة الافتراضية إنجليزي لأن التسميات إنجليزية. لو بدك عربي بدّل lang="ar" ووفّر تسميات عربية.
speak_tts(label, lang="en", slow=False)

cv2.waitKey(0)
cv2.destroyAllWindows()
