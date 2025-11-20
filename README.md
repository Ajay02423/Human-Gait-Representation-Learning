<div align="center">

# 🚶 Human Gait Representation Learning

**Deep Learning Foundation Models for Human Gait Analysis from IMU Sensor Data**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Overview](#-overview) •
[Features](#-features) •
[Installation](#-installation) •
[Usage](#-usage) •
[Architecture](#-architecture) •
[Datasets](#-datasets) •
[Results](#-results) •
[Contributing](#-contributing)

</div>

---

## 🎯 Overview

This project develops **foundation models for human gait representation** using Inertial Measurement Unit (IMU) sensor data. By leveraging self-supervised learning techniques, specifically **Masked Autoencoders (MAE)**, we create robust representations that can be fine-tuned for various downstream tasks.

### 🔬 Applications

- 🦿 **Prosthetic Limb User Training** - Assist in rehabilitation and training
- 🏥 **Rehabilitation** - Support recovery from injuries or paralysis  
- 🎭 **3D Pose Estimation** - Reconstruct human body poses
- 👤 **Person Recognition** - Identity verification through gait patterns
- 🩺 **Gait Disorder Detection** - Identify abnormal walking patterns
- 🎯 **Intention Estimation** - Predict user's movement intentions
- 📏 **Anthropometric Parameter Estimation** - Estimate body measurements

---

## ✨ Features

- ✅ **Self-Supervised Pretraining** using Masked Autoencoder (MAE) architecture
- ✅ **Multi-Dataset Support** - Train on 7+ diverse IMU datasets
- ✅ **Robust Preprocessing Pipeline** - Noise filtering, normalization, and augmentation
- ✅ **Transfer Learning Ready** - Frozen encoder for downstream tasks
- ✅ **Activity Classification** - Fine-tune for HAR (Human Activity Recognition)
- ✅ **Configurable Architecture** - Easy-to-modify hyperparameters
- ✅ **Visualization Tools** - Plot reconstructions and training metrics

---

## 🏗️ Architecture

### Pipeline Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│                 │     │                  │     │                 │
│  IMU Raw Data   │────▶│  Preprocessing   │────▶│  IMU Adapter    │
│  (Acc + Gyro)   │     │  & Filtering     │     │  (128→512×128)  │
│                 │     │                  │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                           │
                                                           ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│                 │     │                  │     │                 │
│  Downstream     │◀────│  MAE Encoder     │◀────│  Random Masking │
│  Tasks          │     │  (Frozen/Tune)   │     │  (75% patches)  │
│                 │     │                  │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### 🧠 Model Architecture

**Masked Autoencoder (MAE) Configuration:**
- **Input**: 512 timesteps × 128 channels (after IMU adapter)
- **Patch Size**: 4
- **Mask Ratio**: 75% (randomly mask patches)
- **Encoder**: 
  - Embedding Dimension: 1024
  - Depth: 24 transformer blocks
  - Attention Heads: 16
- **Decoder**:
  - Embedding Dimension: 512  
  - Depth: 8 transformer blocks
  - Attention Heads: 16

### 📊 Preprocessing Steps

1. **Data Fusion** - Merge accelerometer and gyroscope readings
2. **Resampling** - Interpolate/downsample to 50 Hz
3. **Gravity Filtering** - Remove gravitational component
4. **Noise Cancellation** - Apply Butterworth low-pass filter
5. **Normalization** - Z-score standardization per channel
6. **Windowing** - Segment into fixed-length windows (128 timesteps)

---

## 📚 Datasets

### Self-Supervised Pretraining Datasets

We train on multiple diverse IMU datasets to learn generalizable representations:

| Dataset | Description | Subjects | Activities |
|---------|-------------|----------|------------|
| **CMU-MMAC** | Multimodal activity corpus | 25+ | Cooking, assembly tasks |
| **FLAAP** | Free-living physical activity | 15 | Daily living activities |
| **KUHAR** | Korean dataset | 90 | 18 activities |
| **Mobile-Sensor** | Smartphone sensors | 60 | 6 activities |
| **MMAct** | Multimodal activity dataset | 20 | 36 activities |
| **RealWorld** | Real-world HAR | 15 | 8 activities |
| **WISDM** | Wireless sensor data mining | 51 | 18 activities |

### Downstream Classification Datasets

| Dataset | Task | Classes | Description |
|---------|------|---------|-------------|
| **HARChildren** | Activity Recognition | 6 | Children's activities |
| **Fall Detection-I** | Fall Detection | 2 | Fall vs non-fall |
| **UP-Fall** | Fall Detection | 5 | University fall detection |
| **UMAFall** | Fall Detection | 3 | University of Málaga falls |

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/Ajay02423/Human-Gait-Representation-Learning.git
cd Human-Gait-Representation-Learning

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib scikit-learn scipy pandas
```

---

## 💻 Usage

### 1. Prepare Your Data

Organize your IMU data in the following structure:
```
Data_Processed/
└── Self-supervised_Training_data_arranged/
    ├── dataset1/
    │   ├── subject1_session1.csv
    │   └── subject1_session2.csv
    └── dataset2/
        └── ...
```

Each CSV should contain columns: `acc_x`, `acc_y`, `acc_z`, `gyro_x`, `gyro_y`, `gyro_z`

### 2. Configure Training

Edit `code/config.py` to set your paths and hyperparameters:

```python
class Config_MBM_EEG:
    def __init__(self):
        # Update these paths
        self.root_path = r'/path/to/your/data'
        self.output_path = r'/path/to/output'
        
        # Model hyperparameters
        self.lr = 2.5e-4
        self.batch_size = 128
        self.num_epoch = 10
        self.mask_ratio = 0.75
```

### 3. Train the Model

```bash
cd code

# Self-supervised pretraining
python mae.py --config Config_MBM_EEG

# Fine-tuning for classification
python training.py --pretrained_path /path/to/checkpoint.pth
```

### 4. Visualize Results

```bash
# Visualize reconstructions
python experimentation.py
```

This will generate plots showing:
- Original IMU signals
- Reconstructed signals from the MAE
- Channel-wise comparisons

---

## 📁 Repository Structure

```
Human-Gait-Representation-Learning/
│
├── code/
│   ├── config.py              # Configuration classes for different experiments
│   ├── dataset.py             # IMU dataset loader and preprocessing
│   ├── mae.py                 # Masked Autoencoder model implementation
│   ├── training.py            # Training loops and IMU adapter
│   ├── utils.py               # Utility functions (model loading, etc.)
│   └── experimentation.py     # Visualization and experimentation scripts
│
├── .gitignore                 # Git ignore rules
├── LICENSE                    # MIT License
└── README.md                  # This file
```

---

## 📈 Results

### Classification Performance

After pretraining on 7 diverse IMU datasets and fine-tuning on downstream tasks:

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | ~72% |
| **Test Accuracy** | ~75% |
| **Training Stability** | High (smooth convergence) |

### Activity-wise Performance

- ✅ **Strong**: Walking, Running, Standing
- ⚠️ **Moderate**: Sitting, Laying (some confusion between sedentary activities)
- ✅ **Good Generalization**: Model transfers well to unseen subjects

### Key Findings

- 🎯 Self-supervised pretraining significantly improves downstream performance
- 🔄 75% masking ratio provides optimal reconstruction challenge
- 📊 Multi-dataset pretraining enhances robustness and generalization
- ⚡ Frozen encoder features work well for transfer learning

---

## 🔮 Future Work

- [ ] **Subject Identification** - Biometric recognition via gait patterns
- [ ] **Temporal Segmentation** - Detect activity boundaries automatically  
- [ ] **Advanced Action Recognition** - Recognize complex transitions and object interactions
- [ ] **Real-time Inference** - Optimize for edge deployment on wearables
- [ ] **Multi-modal Fusion** - Integrate with video/depth sensors
- [ ] **Attention Visualization** - Interpret what the model learns

---

## 👥 Contributors

This project is a collaborative effort by:

- **Ajay** - [@Ajay02423](https://github.com/Ajay02423)
- **Sayan**
- **Dipan**  
- **Amuel**
- **Umang**
- **Saksham**

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{humangait2025,
  title={Human Gait Representation Learning: Foundation Models for IMU-based Activity Recognition},
  author={Ajay and Sayan and Dipan and Amuel and Umang and Saksham},
  year={2025},
  publisher={GitHub},
  url={https://github.com/Ajay02423/Human-Gait-Representation-Learning}
}
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 💬 Contact & Support

- 📧 **Issues**: [GitHub Issues](https://github.com/Ajay02423/Human-Gait-Representation-Learning/issues)
- 💡 **Discussions**: [GitHub Discussions](https://github.com/Ajay02423/Human-Gait-Representation-Learning/discussions)
- ⭐ If you find this project helpful, please consider giving it a star!

---

<div align="center">

**Made with ❤️ for advancing human gait analysis research**

[⬆ Back to Top](#-human-gait-representation-learning)

</div>
