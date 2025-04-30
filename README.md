# Music Genre Classification

**Report Link:**  
(https://docs.google.com/document/d/1GRdB4p3Wy2gIoguUn2gJcUwLOuCMR_ttBRAXQtfMpjg/edit?tab=t.0#heading=h.ab53ba3frjds)

**Presentation Link:**  
[Insert presentation URL here]

---

## Introduction

This project explores several approaches to improve music genre classification on the GTZAN Genre Collection (1,000 audio clips across 10 genres). We compare different:

- **Input representations:** spectrograms, raw waveforms, and pre-extracted musical features  
- **Model architectures:** custom neural networks (including CNN, LSTM) and pre-trained models  
- **Data augmentation techniques**  
- **Training set sizes** (learning-curve and bootstrap analyses)

Our goal is to identify which combinations yield the best classification accuracy and robustness.

---

## Dataset

**Note:** The raw audio files are not included in this repository due to size and licensing. All team members can access the private GTZAN dataset via Kaggle.

### Download Instructions

1. **Obtain your Kaggle API key**  
   - Log in to [Kaggle](https://www.kaggle.com/account).  
   - Under **API**, click **Create New API Token** to download `kaggle.json`.  

2. **Configure your environment**  
   ```bash
   mkdir -p ~/.kaggle
   mv /path/to/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
3. **Download Dataset:**  
   ```bash
    bash dataset_downloader.sh
   ```


# Folder Structure

├── 100_data/  
│   ├── features_30_sec.csv  
│   └── preprocessed_data.npz  
│
├── 200_code/  
│   ├── 201_processing/           # audio augmentation & spectrogram generation  
│   ├── 202_spec_modelling/       # spectrogram-based model code  
│   ├── 203_waveform_modelling/   # raw waveform model code  
│   └── 204_feature_modelling/    # pre-extracted feature model code  
│
├── models/                       # saved model checkpoints  
├── outputs/                      # figures, logs, and evaluation metrics  
├── dataset_downloader.sh         # downloads raw audio via Kaggle API  
└── requirements.txt              # Python dependencies  

# Running Instructions
