language : [english](https://github.com/dodo701/voiceger-for-cloud/blob/main/README.md) [korean](https://github.com/dodo701/voiceger-for-cloud/blob/main/README_ko.md)

# Voiceger For cloud services!

This repository provides an removed streamlit and safetensors inference version of voiceger V2. The mission is to remove overhead and maximizing tts inference speed on cpu.
This repository is the fork of voiceger-v2, which created by zunzun999 and yh00072.

## why this fork?

1. voiceger v2 uses very cpu-demanding models(cnhubert-roberta,bert,gpt and sovits), the cpu-only inference speed is extremely low.
2. this repository has not been updated since 2025. i'll modernize the dependency by applying upstream changes and enhance efficiency.
3. also, i will package this repository by docker. whisch allows to be easily deployed on cloud developement or servers.

## Overview

This project is based on [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) and has been adapted and fine-tuned for Zundamon's voice synthesis.

## Features

1. Has CLI Tools for fast synthesis.
2. Emotion/style presets for TTS: select a Zundamon emotion, enter text, and generate audio; multilingual input is supported.
3. converted BERT and Hubert model with **safetensors** format and GPT and SoVITS model with **ONNX** for faster CPU inference. 

## Prerequisites

Before starting, ensure you have the required dependencies installed:

```bash
pip install -r requirements_final.txt
```
- Notice: You have to install Cmake since pyopenjtalk requires to build wheel.
- If you don't use any GPU for inference, install torch with extra index.
  

After installing the dependencies, install PyTorch and its companion packages. Both `torchvision` and `torchaudio` are required.

The following versions are tested and verified to work successfully:

- **PyTorch and torchAudio**: `2.11.0+cpu`
- **CUDA**: removed.
- **Python**: `3.9`
- **onnx with OPENvino**: `1.24.1`
- **transformers**

## Setup Instructions

### Step 1: Clone the Repository

To ensure all required submodules are initialized properly, use the following command:

```bash
git clone https://github.com/dodo701/voiceger_v2.git
cd voiceger_v2
```

### Step 2: Download Pretrained Models for TTS function

**Download GPT-SoVITS Pretrained Models and Zundamon finetuned models**
    - Use the following commands to download and set up the models:
        
        ```bash
        (enable venv in python.)
        python download.py
        ```
    
    -This script will automatically download **all** models in right directories.
    -for more information about the models, visit [my huggingface repository!](https://huggingface.co/Geukrak/onnx_zundamon):
      

#### Install ffmpeg(not needed actually.)

no ffmpeg install needed! (this software uses soundfile and soxr to process audio.)

#### Fine-Tuned Models
If you have fine-tuned indices and weights, place:
- Indices into `voiceger-webui/GPT-SoVITS/Retrieval-based-Voice-Conversion-WebUI/assets/indices`
- Weights into `voiceger-webui/GPT-SoVITS/Retrieval-based-Voice-Conversion-WebUI/assets/weights`
- Indices download URL:
https://huggingface.co/zunzunpj/zundamon_RVC/blob/main/zumdaon_rvc_indices_weights/train-0814-2_IVF256_Flat_nprobe_1_train-0814-2_v2.index
- Weights download URL:
https://huggingface.co/zunzunpj/zundamon_RVC/blob/main/zumdaon_rvc_indices_weights/train-0814-2.pth

### Additional Requirements for Windows Installation

1. **Install Visual Studio Build Tools**
    - Visit the [Visual Studio Download Page](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
    - Download and install "Visual Studio Build Tools".
    - During installation, select "Desktop development with C++".
2. **Install CMake**
    - Visit the [CMake Official Site](https://cmake.org/download/).
    - Download and install the Windows version of CMake.
    - During installation, choose "Add CMake to the system PATH".

### Troubleshooting

If you encounter an error like:

```
An error occurred during inference:
Resource averaged_perceptron_tagger_eng not found.
```

Try running the following commands in your project environment:

```python
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
```
This will ensure the necessary NLTK resources are downloaded.

For more installation information of pretrained models, please refer to the following resources:
- [GPT-SoVITS Official Repo](https://github.com/RVC-Boss/GPT-SoVITS)


## How to Use the WebUI(deprecated.)
since this repository is created for fast voice processing in non-gpu server, webui will not be supported. use infer_tts_minimal.py for substitution.

I will soonly create a new branch with webui is supported.

## How to Generate Zundamon's Voice Audio

1. Select Emotion
    - Pick an emotion preset from the UI.
2. Language Selection
    - Choose the language that matches your text.
3. Enter Target Text
    - Type the text you want Zundamon to speak.
4. Generate Audio
    - Click the **Generate Speech** button. After processing, preview the audio and use the download button to save the result.

## License Information

This software includes the following open-source software:

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) (MIT License)
- [GPT-SoVITS Pretrained Models](https://huggingface.co/lj1995/GPT-SoVITS) (MIT License)
- [UVR5 (Voice Cleaning)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/uvr5_weights) (MIT License)
- [Faster Whisper Large V3](https://huggingface.co/Systran/faster-whisper-large-v3) (MIT License)
- [RVC WebUI Official Repo](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) (MIT License)
- [RMVPE](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt) (MIT License)

These are provided under their respective license terms.

The license for the Zundamon Voice model is as follows:
https://zunko.jp/con_ongen_kiyaku.html


