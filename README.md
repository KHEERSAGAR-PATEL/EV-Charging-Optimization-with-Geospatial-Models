# EV-Charging-Optimization-with-Geospatial-Models
---

# 🚀 Intel-Optimized Machine Learning Suite for Real-World Performance

A powerful, production-focused suite of **Intel®-accelerated ML projects** showcasing classical machine learning, deep learning, and reinforcement learning pipelines. Optimized from the ground up for **speed, scalability, and deployment**—these solutions reflect industry-grade engineering and domain versatility.

> 🧠 Designed to impress Applied Scientist and ML Engineer recruiters at top-tier companies (MAANG+), with reproducible code, benchmarked performance, and hardware-aware implementation.

---

## 🔍 Project Portfolio Overview

### 1. 🧠 Deep Learning Classifier (Intel Optimized)
- **Model**: Custom ConvNeXt-style CNN
- **Dataset**: Intel Scene Classification Dataset (Natural Images)
- **Highlights**:
  - Achieved **58.7% Top-1 Accuracy**
  - **Intel Extension for TensorFlow (ITEX)** + `jit_compile=True`
  - End-to-end graph optimization with XLA
  - ~40% reduction in training time

---

### 2. 🎯 Classical ML Classifiers (Intel scikit-learn)
- **Goal**: Benchmark core ML models with Intel-optimized Scikit-learn
- **Models Used**: Logistic Regression, SVM, Random Forest, k-NN, XGBoost, Decision Tree
- **Highlights**:
  - Leveraged `scikit-learn-intelex` for 20–45% speedup
  - Benchmarked **8+ classifiers** on custom tabular datasets
  - Achieved **~93.7% accuracy** on XGBoost with ~14 ms/sample latency
  - Built for edge and CPU-heavy production environments

---

### 3. 🧠 Reinforcement Learning Agent (Intel RLlib)
- **Goal**: Solve CartPole-v1 using policy-based learning
- **Framework**: Ray + ITEX + TensorFlow
- **Model**: PPO (Proximal Policy Optimization)
- **Highlights**:
  - Reward threshold (200) achieved in < 80 episodes
  - ITEX acceleration with `TFPolicyGraph`
  - ~1.7x training time reduction with Intel-optimized ops

---

## ⚡️ Deep Dive: Classical ML Classifiers

> A specialized Intel-optimized notebook for robust, scalable classification pipelines using traditional ML.

### ✅ Key Features
- Intel® Extension for Scikit-learn for **parallelized CPU acceleration**
- Thorough pipeline: preprocessing, model selection, grid search, CV, and evaluation
- Real-world relevance in fraud detection, diagnostics, and marketing

### 🧪 Benchmarked Models

| Model               | Optimization         | Notes                                |
|--------------------|----------------------|--------------------------------------|
| Logistic Regression| Intel Extension      | Fast and efficient linear modeling   |
| K-Nearest Neighbors| Intel Extension      | Optimized for memory & speed         |
| Random Forest      | Intel Extension      | 3x faster with identical accuracy    |
| XGBoost            | Native + Tuned       | Tabular powerhouse, optimized trees  |
| Decision Tree      | Intel Extension      | Lightweight and interpretable        |
| SVM (Linear, RBF)  | Intel Extension      | Vectorized kernel ops, faster Cython |
| Gradient Boosting  | Native/Sklearnex     | Boosted accuracy, hardware-tuned     |

### 📈 Performance Snapshot

| Dataset        | Best Model     | Accuracy | F1 Score | Inference Time |
|----------------|----------------|----------|----------|----------------|
| Tabular (custom)| XGBoost        | 93.7%    | 92.1%    | ~14 ms/sample  |
| Benchmark      | Random Forest  | 91.2%    | 90.4%    | ~10 ms/sample  |
| Linear Baseline| Logistic Regr. | 89.5%    | 88.7%    | ~8 ms/sample   |

---

## 🏗️ Tech Stack

- **Intel AI Analytics Toolkit**
  - ITEX (TensorFlow Extension)
  - scikit-learn-intelex
  - Intel Distribution for Python
- **Frameworks**
  - TensorFlow 2.x, Scikit-learn, XGBoost, Ray RLlib
- **Languages/Libraries**
  - Python, Pandas, NumPy, Matplotlib, ONNX
- **Optimization Tools**
  - XLA, OneDNN Graph, JIT Compilation



## 📁 Repo Structure

```
intel-ml-optimized-suite/
├── intel_optimized_deep_learning_classsifier.ipynb
├── intel_optimized_sklearn_classifiers.ipynb
├── intel_optimized_reinforcement_learning.ipynb
├── README.md
└── requirements.txt
```

---

## 📊 Performance Summary

| Project        | Optimizations Used         | Speedup | Accuracy / Reward |
|----------------|----------------------------|---------|-------------------|
| Deep Learning  | ITEX + XLA + OneDNN        | ~1.4x   | 58.7% Top-1       |
| Classical ML   | scikit-learn-intelex       | ~3x     | 81–94% Accuracy   |
| RL Agent       | ITEX + TFPolicyGraph + Ray | ~1.7x   | Avg Reward: 200   |

---

## 💼 Why This Matters for MAANG

✅ **Production-Ready**: Hardware-aware, efficient, scalable pipelines  
✅ **Versatile**: Covers classical ML, DL, and RL use-cases  
✅ **Experiment-Driven**: Comparative plots, logging, and reproducibility  
✅ **Deployable**: Suitable for edge, low-latency and high-throughput systems  
✅ **Well-Engineered**: Follows best practices in modularity, performance, and explainability

---



## ❤️ Made with Love by *Kheer Sagar Patel* 
*M.Tech in AI & ML, IIITDM Jabalpur*  
> Blending research depth with real-world engineering.  
> Building ML systems that are elegant, optimized, and impactful.


> "Optimizing for performance isn't a step—it's a mindset."

---
