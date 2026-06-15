# 🚗 Classifier de Vehicule prin Deep Learning (ResNet-18)

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-98%25-success?style=for-the-badge)](https://github.com/)

Acest proiect conține o implementare completă a unui classifier de imagini pentru **20 de tipuri diferite de vehicule**, utilizând arhitectura **ResNet-18** pre-antrenată în PyTorch și adaptată prin tehnici de *transfer learning* (fine-tuning). Modelul final obține o acuratețe remarcabilă de **98%** pe setul de validare.

Proiectul a fost dezvoltat ca parte a activității academice pentru disciplina **Securitatea Informației (SI)**, Facultatea de Automatică și Calculatoare (Anul 3, Semestrul 2).

---

## 🌟 Caracteristici Cheie

* **Arhitectură de Top:** Fine-tuning pe **ResNet-18** cu deblocarea tuturor straturilor (`requires_grad = True`) pentru o adaptare optimă la specificul vehiculelor.
* **Regularizare:** Integrarea unui strat de `Dropout(0.4)` înainte de clasificatorul liniar final pentru a preveni overfitting-ul.
* **Rată de Învățare Diferențiată (Differential Learning Rates):**
  * `1e-4` pentru extractorul de caracteristici (straturile pre-antrenate ResNet)
  * `1e-3` pentru capul de clasificare final (`model.fc`)
* **Tratare Robustă a Datelor:** Implementarea clasei personalizate `SafeImageFolder` care moștenește `ImageFolder` din torchvision, concepută special pentru a intercepta și sări peste imaginile corupte (`UnidentifiedImageError` / `OSError`) fără a întrerupe procesul de antrenare sau evaluare.
* **Augmentarea Datelor:** Transformări avansate de tip: rotații aleatoare, oglindiri (flips) și ajustări de contrast/luminozitate (color jittering) pentru a spori diversitatea setului de antrenare.
* **Interfață Interactivă:** Widget interactiv (`ipywidgets`) încorporat în Jupyter Notebook pentru încărcarea unei imagini proprii și obținerea predicției modelului în timp real, alături de procentul de încredere (Confidence Score).

---

## 📂 Structura Setului de Date

Setul de date este organizat în directorul `dataset/`, având subdirectoare specifice pentru fiecare clasă în parte (format standard `ImageFolder`):

```text
dataset/
├── airplane/
├── ambulance/
├── bicycle/
├── boat/
├── bus/
├── car/
├── fire_truck/
├── helicopter/
├── hovercraft/
├── jet_ski/
├── kayak/
├── motorcycle/
├── rickshaw/
├── scooter/
├── segway/
├── skateboard/
├── tractor/
├── truck/
├── unicycle/
└── van/
```

### Statistici Set de Date:
* **Număr total clase:** 20 de tipuri de vehicule.
* **Număr total imagini:** 3680 de imagini.
* **Split antrenare/validare:** 80% antrenare (2944 imagini) | 20% validare (736 imagini).
* **Rezoluție imagini input:** Redimensionate la $224 \times 224$ pixeli.

---

## 🛠️ Instalare și Configurare

Pentru a rula notebook-ul local, urmează pașii de mai jos:

### 1. Clonarea Depozitului
```bash
git clone https://github.com/utilizator/Proiect_DeepLearning_Masini.git
cd Proiect_DeepLearning_Masini
```

### 2. Crearea și Activarea unui Mediu Virtual (Recomandat)
Pe Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Instalarea Dependențelor
Instalează bibliotecile necesare rulării modelului:
```bash
pip install torch torchvision matplotlib scikit-learn numpy tqdm pillow ipywidgets
```

*Notă: Dacă ai o placă grafică dedicată NVIDIA, asigură-te că instalezi versiunea de PyTorch cu suport CUDA corespunzătoare pentru accelerare GPU.*

---

## 🚀 Utilizare

### Antrenarea și Evaluarea Modelului
Deschide Jupyter Notebook și rulează celulele în ordine:
```bash
jupyter notebook vehicle_classifier.ipynb
```

În notebook se parcurg următoarele etape:
1. Verificarea disponibilității GPU-ului (CUDA).
2. Încărcarea setului de date prin `SafeImageFolder` și aplicarea transformărilor.
3. Inițializarea arhitecturii ResNet-18 cu ponderile pre-antrenate implicit și definirea optimizatorului Adam cu rate de învățare diferențiate.
4. Antrenarea modelului timp de 10 epoci cu urmărirea acurateții și a funcției de cost (Loss).
5. Încărcarea stării salvate a modelului (`model.pth`).
6. Generarea raportului de clasificare complet.

---

## 📊 Rezultate și Performanțe

Modelul obține performanțe excepționale pe setul de validare (736 de imagini):

* **Acuratețe Globală (Accuracy):** **98%**
* **Macro Average F1-score:** **98%**

### Raport Detaliat de Clasificare (Classification Report)

| Clasă | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| ✈️ **airplane** | 0.97 | 1.00 | 0.98 | 29 |
| 🚑 **ambulance** | 1.00 | 0.97 | 0.99 | 35 |
| 🚲 **bicycle** | 0.92 | 0.96 | 0.94 | 23 |
| ⛵ **boat** | 1.00 | 0.98 | 0.99 | 42 |
| 🚌 **bus** | 0.95 | 1.00 | 0.97 | 36 |
| 🚗 **car** | 1.00 | 0.98 | 0.99 | 46 |
| 🚒 **fire_truck** | 0.97 | 0.90 | 0.94 | 40 |
| 🚁 **helicopter** | 1.00 | 1.00 | 1.00 | 39 |
| 🚤 **hovercraft** | 0.97 | 0.97 | 0.97 | 36 |
| 🌊 **jet_ski** | 0.93 | 1.00 | 0.96 | 27 |
| 🛶 **kayak** | 0.97 | 0.95 | 0.96 | 38 |
| 🏍️ **motorcycle** | 1.00 | 1.00 | 1.00 | 33 |
| 🛺 **rickshaw** | 1.00 | 1.00 | 1.00 | 41 |
| 🛴 **scooter** | 0.97 | 1.00 | 0.99 | 38 |
| 🛹 **segway** | 0.96 | 0.96 | 0.96 | 53 |
| 🛹 **skateboard** | 1.00 | 0.97 | 0.98 | 32 |
| 🚜 **tractor** | 1.00 | 1.00 | 1.00 | 39 |
| 🚛 **truck** | 0.94 | 1.00 | 0.97 | 33 |
| 🚲 **unicycle** | 0.98 | 0.95 | 0.96 | 42 |
| 🚐 **van** | 1.00 | 1.00 | 1.00 | 34 |

---

## 🖼️ Widget-ul de Testare Interactivă

La finalul notebook-ului, este disponibil un widget intuitiv realizat cu ajutorul `ipywidgets`:

1. Apasă pe butonul **"Incarca Poza"** (Upload).
2. Selectează o imagine de vehicul din calculatorul tău.
3. Widget-ul va afișa automat imaginea scalată și va printa predicția modelului împreună cu procentajul de siguranță:

```text
------------------------------
Model Prezis: CAR
Nivel de Incredere: 99.45%
------------------------------
```

---

## 💾 Structura Fișierelor Salvate

Modelul optim este salvat sub forma unui fișier checkpoint numit `model.pth`. Acesta conține un dicționar cu următoarele elemente cheie:
* `model_state_dict`: Starea parametrilor și ponderilor rețelei antrenate.
* `class_names`: Lista celor 20 de clase identificate în setul de date.
* `num_classes`: Numărul total de clase (20).

Pentru a încărca modelul în alte scripturi externe de Python:
```python
import torch
from torchvision import models

# Reconstrucția arhitecturii modelului
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.4),
    torch.nn.Linear(model.fc.in_features, 20)
)

# Încărcarea ponderilor salvate
checkpoint = torch.load("model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()
```
