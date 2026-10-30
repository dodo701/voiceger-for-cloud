언어 : [영어](https://github.com/dodo701/voiceger-for-cloud/blob/main/README.md) [한국어](https://github.com/dodo701/voiceger-for-cloud/blob/main/README_ko.md)

# 클라우드로 돌아가는 voiceger 프로그램.

하지만 빨랐죠? 가 중요한 클라우드 실시간 tts 파이프라인을 위해 만들었습니다.
아직까지 업데이트와 기능 안정화가 진행 중이예요!

## 그래서 이걸 왜 만드셨습니까...

1. voiceger v2 원본 프로그램은 GPU가 있어야 사실상 실시간 렌더링이 가능함. 나같은 cpu밖에 없는 거지들은 못 돌림!!
2. 원본 레포지토리는 2025년부터 업데이트가 멎음. 대부분의 디펜던시와 모델 로드 작업을 극단적으로 최적화할 거임.
3. 이 레포 전체를 도커로 패키징해서, 서버 배포를 쉽게 할 거임!!!

## 프로젝트 개요

이 레포는 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 을 기반으로 한국어, 영어, 일본어의 즌다몬 음성 합성을 진행하기 위해 만들어짐.

## 기능들!

1. 기본 cli 기반 음성 합성 툴 존재.(infer_tts_minimal.py)
2. Emotion/style presets for TTS: select a Zundamon emotion, enter text, and generate audio; multilingual input is supported.
3. 

## Prerequisites

Before starting, ensure you have the required dependencies installed:

```bash
pip install -r requirements_final.txt
```
- Notice: You have to install Cmake since pyopenjtalk requires to build wheel.
- If you don't use any GPU for inference, install torch with extra index.
  

After installing the dependencies, install PyTorch and its companion packages.`torchaudio` are required.

The following versions are tested and verified to work successfully:

- **파이토치와 토치 오디오**: `2.11.0+cpu`
- **쿠다**: removed.
- **비단뱀**: `3.11.9`
- **인텔 오픈비노 오닉스 런타입**: `1.24.1`
- **허깅패이스 트랜스포머**

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


                                                                                                 
