# Derin Öğrenme Projesi: ResNet ve YOLOv8 ile Hibrit Model

Bu proje, derin öğrenme alanında **ResNet** ve **YOLOv8** modellerini birleştirerek oluşturulan hibrit bir modeli ve performansını kapsamaktadır. Aşağıda, modelin oluşturulma aşamaları, parametre ayarları ve çeşitli çıktılar açıklanmaktadır.

---

## 📚 ResNet Modelinin Oluşturulması
**ResNet (Residual Network)**, görüntü sınıflandırma için kullanılan bir derin öğrenme modelidir. Bu projede, ResNet modeli şu şekilde oluşturulmuştur:

### 🔧 Parametreler:
- **Model Yapısı**: ResNet-50
- **Optimizasyon**: Adam Optimizer (learning rate: 0.001)
- **Loss Fonksiyonu**: Cross-Entropy Loss
- **Batch Size**: 32
- **Epoch Sayısı**: 50
- **Veri Artırma**: Random crop, flip, rotation

Kod Örneği:
```python
from torchvision.models import resnet50
import torch.optim as optim

model = resnet50(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()
```

### 📈 ResNet TensorBoard Çıktısı:
Modelin eğitimi sırasında elde edilen kayıp (loss) ve doğruluk (accuracy) değerleri TensorBoard ile görselleştirilmiştir. Aşağıda bir örnek görüntü yer almaktadır:


![Tensorboard Çıktısı]![image](https://github.com/user-attachments/assets/a22bbaa8-b22e-499b-868e-bd369825ae03)


---

## 🎯 Doğruluk Değerleri
- **Eğitim Doğruluğu**: %98.3
- **Doğrulama Doğruluğu**: %95.6

---

## 🖼️ Gradio Arayüzü
Model, kullanıcıların el hareketlerini test edebilmesi için bir **Gradio** arayüzüyle entegre edilmiştir. Bu arayüz, görsel olarak basit ve kullanımı kolaydır.

### Örnek Çıktı:
![Gradio Çıktısı]![image](https://github.com/user-attachments/assets/c875fe84-cab6-415d-a4b2-7ece361c6a5a)


---

## 🤝 Hibrit Model: ResNet + YOLOv8
**YOLOv8** modeli, nesne tespiti için kullanılmış ve ResNet-50 ile birleştirilerek daha güçlü bir hibrit model oluşturulmuştur.

### 📐 Çalışma Şekli:
1. **YOLOv8**: Görüntüdeki nesneleri tespit eder.
2. **ResNet-50**: Tespit edilen nesneleri sınıflandırır.
3. **Birleştirme**: Tespit ve sınıflandırma sonuçları birleştirilerek nihai çıktı oluşturulur.

Kod Örneği:
```python
from ultralytics import YOLO

yolo_model = YOLO('yolov8.pt')
detections = yolo_model(image)

for obj in detections:
    cropped_obj = crop_image(image, obj.bbox)
    resnet_prediction = resnet_model(cropped_obj)
    print(resnet_prediction)
```

### 🧪 Hibrit Model Çıktısı
Hibrit modelin test sonuçları:
- **Toplam Doğruluk**: %97.8
- **İşlem Süresi**: Ortalama 0.12 saniye/görüntü

---

## 🔍 Grad-CAM Görselleştirmesi
Grad-CAM (Gradient-weighted Class Activation Mapping) yöntemi, modelin karar verirken hangi alanlara odaklandığını görselleştirmek için kullanılmıştır.

### Örnek Çıktı:![WhatsApp Image 2025-02-05 at 14 33 46_c3bceaac](https://github.com/user-attachments/assets/ccbd2d2d-3304-4267-a945-2fa7bd0613e5)

![Grad-CAM Çıktısı]()

---

## Özet
Bu proje, ResNet ve YOLOv8 modellerini birleştirerek oluşturulan hibrit yapısıyla görüntü sınıflandırma ve nesne tespiti alanında yüksek performanslı bir çözüm sunmaktadır. TensorBoard, Grad-CAM ve Gradio gibi araçlarla kullanıcı deneyimi geliştirilmiş ve modelin doğruluğu artırılmıştır.

