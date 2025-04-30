# Music Genre Classification

![Music Genre Illustration](https://music.vinetogo.com/wp-content/uploads/2024/07/music-genre.jpg)

## Links to Reports

**Report Link:**  
[Music Genre Classification Report](https://docs.google.com/document/d/1GRdB4p3Wy2gIoguUn2gJcUwLOuCMR_ttBRAXQtfMpjg/edit?tab=t.0#heading=h.ab53ba3frjds)

**Presentation Link:**  
[Music Genre Classification Presentation](https://docs.google.com/presentation/d/1jcWxvFa9KaiItuE7CHC1lh-snfG9U-EIDXSOQP_95gM/edit?slide=id.g351aa865629_0_88#slide=id.g351aa865629_0_88)

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
	•	100_data/
	•	features_30_sec.csv
	•	preprocessed_data.npz
	•	200_code/
	•	201_processing/ — Audio augmentation & spectrogram generation
	•	202_spec_modelling/ — Spectrogram-based model code
	•	203_waveform_modelling/ — Raw-waveform model code
	•	204_feature_modelling/ — Pre-extracted-feature model code
	•	300_models/ — Saved model checkpoints and best models
	•	dataset_downloader.sh — Script to download raw audio via the Kaggle API
	•	requirements.txt — Python dependencies

## Running Analysis and Feature Extraction
In order to run the files user need to download necessary libraries. It can be achieved by running following command

```bash
pip install -r requirements.txt
```

All Jupyter notebooks (`*.ipynb`) are designed to reproduce our study results when run in order. **Before** training any models, execute the scripts in the **201_processing/** folder to generate the necessary preprocessed data.

### Feature Extraction

The `feature_extractor.py` script in **201_processing/** defines an `AudioFeature` class that replicates the same feature set used in our experiments. To extract features for any `.wav` file and obtain a pandas DataFrame:

```python
from feature_extractor import AudioFeature

test_input_path = 'path/to/audio.wav'
audio_features = AudioFeature(test_input_path)
df = audio_features.get_dataframe()
```


## Loading Pretrained Models

You can skip training and load our custom models directly from the `300_model/` folder.

### Load the Fine-Tuned XGBoost Model

```python
from xgboost import XGBClassifier

xgb_model = XGBClassifier()
xgb_model.load_model('300_model/xgb_tuned_model.json')
```

### Load the customly created  Neural Network

```python
from tensorflow.keras.models import load_model

nn_model = load_model('300_model/best_neural_network_model_30sec.h5')
```