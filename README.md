# DeCa: De-bias Incremental Detection via Causal Intervention

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-1.9+-red.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)
![Framework](https://img.shields.io/badge/framework-MMDetection-orange.svg)

**DeCa** is a PyTorch-based framework for **domain incremental object detection**, designed to handle evolving data distributions over time (e.g., changing domains, weather conditions, or image styles).  
By integrating *causal intervention* and *adaptive memory mechanisms*, DeCa effectively extracts domain-invariant features while mitigating catastrophic forgetting.

---

## 🚀 Features

- **Unified Feature Memory (UFM):** Maintains diverse and representative embeddings across domains to retain historical knowledge.  
- **Causal Intervention Module (CIM):** Applies attention-based causal correction to disentangle confounding factors.  
- **Built upon MMDetection:** Fully compatible with [MMDetection](https://github.com/open-mmlab/mmdetection) for efficient integration and extensibility.  
- **Cross-domain robustness:** Demonstrated effectiveness across weather, style, and dataset domain shifts.  

---

## 🧩 Installation

```bash
# Clone this repository
git clone https://github.com/libo-huang/DeCa.git
cd DeCa

# (Recommended) Create a conda environment
conda create -n deca python=3.10 -y
conda activate deca

# Install dependencies
pip install -r requirements.txt

# Install MMDetection dependencies
# See: https://mmdetection.readthedocs.io/en/latest/install.html
```

## 🧪 Usage
### 🏋️ Training

1. Prepare your dataset(s) and domain-incremental splits (e.g., Clear → Fog → Rain).
2. Modify the configuration file in configs/.
3. Start training:
``` bash
python tools/train.py configs/deca_config.py
```
For incremental learning, load the previous checkpoint and continue training:
``` bash
python tools/train.py configs/deca_config.py --resume-from work_dirs/prev_checkpoint.pth
```

### 📈 Evaluation
``` bash
python tools/test.py configs/deca_config.py work_dirs/deca_latest.pth --eval mAP
```

### 🎨 Demo
Visualize predictions under different domains:
``` bash
python demo/demo_incremental.py configs/deca_config.py work_dirs/deca_latest.pth demo/input.jpg
```
Example outputs are saved in demo/results/.



## 📂 Directory Structure
``` bash
DeCa/
├── configs/             # Config files for training & evaluation
├── demo/                # Demo scripts and sample images
├── docs/                # Documentation and experiment details
├── mmdet/               # Modified MMDetection modules
├── tools/               # Training & evaluation utilities
├── requirements.txt     # Dependencies
├── LICENSE              # Apache 2.0 License
└── README.md            # Project description
```

## 🧠 Key Idea

DeCa integrates causal inference into object detection to enhance robustness against environmental shifts.

- The UFM preserves representative domain features.

- The CIM performs causal correction through attention, computing the expected causal effect $𝑃(𝑦∣do(𝑥))$.
Together, they enable stable, cross-domain detection performance under incremental adaptation.



## 🧑‍💻 Contributing
We welcome community contributions!
To contribute:
1. Fork this repository
2. Create your feature branch (git checkout -b feature/new-feature)
3. Commit your changes (git commit -m 'Add new feature')
4. Push to the branch (git push origin feature/new-feature)
5. Open a Pull Request

## 🙏 Acknowledgements

This implementation is built upon the excellent [MMDetection](https://github.com/open-mmlab/mmdetection) framework by OpenMMLab.
We sincerely thank the MMDetection community for providing a powerful and flexible foundation for modern object detection research.

## 📜 License

This project is released under the Apache License 2.0.

