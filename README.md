# SA-AOT-GAN: Structure-Aware Image Inpainting with Soft Edge Supervision

## 📌 Overview

This project proposes **SA-AOT-GAN (Structure-Aware AOT-GAN)**, a lightweight structure-aware image inpainting framework designed to improve structural consistency in large missing regions.

Built upon the strong baseline **AOT-GAN**, our method introduces a **lightweight Edge Head** and a **soft edge supervision strategy** to explicitly guide structural learning during training, while **keeping inference cost unchanged**.

---

## 🚀 Key Features

- Structure-aware learning via Edge Head
- Soft edge supervision (Sobel + L1 loss)
- Training-only auxiliary branch (no inference overhead)
- Lightweight design
---

## 🧠 Method Overview

Encoder → AOT Blocks → Decoder  
        ↓  
      Edge Head (training only)

---

## 🔍 Soft Edge Supervision

- Sobel gradient magnitude used as edge label
- Continuous values instead of binary edges
- L1 loss for stable training


---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/SA-AOT-GAN.git
cd SA-AOT-GAN
pip install -r requirements.txt
```

---

## 🏋️ Training

```bash
python train.py --edge_weight 0.3
```

---

## 🧪 Testing (Fair Evaluation)

```bash
python test.py \
  --pre_train path/to/model.pt \
  --mask_ratio_min 0.5 \
  --mask_ratio_max 0.8 \
  --mask_seed 2026 \
  --assignment_path experiments/fair_eval/mask_assign_05_08.json \
  --outputs outputs/result
```
