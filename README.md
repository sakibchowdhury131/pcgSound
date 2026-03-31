# pcgSound

A deep learning pipeline for audio/sound classification using spectrogram-based image representations and convolutional neural networks with attention mechanisms.

## Overview

This project converts raw audio signals into spectrograms (2D image representations) and trains a CNN model with channel and pixel attention modules to classify audio samples. The architecture is suited for tasks such as heart sound classification.

## Features

- Audio-to-spectrogram conversion (3D spectrograms, standard spectrograms)
- Custom CNN architecture with **Channel Attention** and **Pixel Attention** modules
- Modular pipeline: data loading, preprocessing, model training, and visualization
- Built on TensorFlow / Keras

## Project Structure

| File | Description |
|------|-------------|
| `audio2img.py` | Converts audio files to spectrogram image representations |
| `models.py` | CNN model with channel and pixel attention layers |
| `config.py` | Configuration entry point |
| `sysModConfig.py` | System/model configuration parameters |
| `fileLoad.py` | Audio file loader |
| `load.py` | Dataset loader |
| `__3dSpec.py` | 3D spectrogram generation |
| `__createSpectrogram.py` | Spectrogram creation utilities |
| `__makeSpec.py` | Spectrogram building helpers |
| `__run.py` | Main pipeline entry point |
| `_visual_spec.py` | Spectrogram visualization |

## Requirements

```
tensorflow
keras
numpy
librosa
```

## Usage

1. Place training and validation audio archives (`training.zip`, `validation.zip`) in the project root.
2. Run the main pipeline:

```bash
python __run.py
```

## Model Architecture

The CNN uses:
- **Pixel Attention**: Conv2DTranspose upsampling + sigmoid gate applied spatially
- **Channel Attention**: GlobalAveragePooling → Dense → sigmoid reweighting of channels
- Batch Normalization after each attention block

## License

MIT
