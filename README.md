# Tacotron 2 on Hábrók — Speech Synthesis

This project customizes the [NVIDIA Tacotron 2 implementation](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2) for training and inference on the Hábrók cluster as part of the Voice Technology MSc. 2024–2025 Speech Synthesis 2 course.

## Original Repository
The full codebase can be obtained by cloning the official NVIDIA DeepLearningExamples repository:

git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2

csharp
Copy
Edit

## Project Description
This project focuses on training and fine-tuning the Tacotron 2 acoustic model and performing speech synthesis using the WaveGlow vocoder. It includes tasks such as:
- Synthesizing custom speech with pre-trained checkpoints
- Training Tacotron 2 on the LJSpeech dataset for 250 epochs
- Fine-tuning the model on the CMU Arctic bdl dataset
- Performing an ablation study by removing the convolutional post-net to observe its impact on speech quality

## What This Repository Contains
This folder only includes modified or newly created files relevant to the project.  
To access the complete codebase, clone the repository from the link above.

Key changes in this project include:
- Adjustments to scripts for training and inference on the Hábrók cluster
- Added configurations for correct synthesis of custom sentences
- Modifications for fine-tuning Tacotron 2 on a small dataset (CMU Arctic bdl)
- Changes to remove the post-net module for the ablation study

## How to Run

### 1. Clone the Original Repository
```bash
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2
### 2. Set Up the Environment
```bash
python3 -m venv tacotron_env
source tacotron_env/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
pip install torch
3. Download Pre-trained Checkpoints
Download both Tacotron 2 and WaveGlow checkpoints and place them in the Tacotron 2 directory.

4. Perform Inference
bash
Copy
Edit
python inference.py --tacotron2 <path_to_tacotron2_checkpoint> \
                    --waveglow <path_to_waveglow_checkpoint> \
                    --wn-channels 256 \
                    -o output/ \
                    -i phrases/phrase.txt \
                    --fp16
Replace <path_to_tacotron2_checkpoint> and <path_to_waveglow_checkpoint> with the correct paths.

5. Train the Model
To train Tacotron 2 on the LJSpeech dataset:

bash
Copy
Edit
bash scripts/prepare_dataset.sh
bash scripts/train_tacotron2.sh
6. Fine-tune on CMU Arctic bdl Dataset
Follow the same training process but point to the CMU Arctic bdl dataset and use the LJSpeech-trained checkpoint as the starting point.

7. Perform the Post-Net Ablation Study
Modify the Tacotron 2 architecture by removing the convolutional post-net and retrain the model to compare results.

Language
The project was developed using English datasets (LJSpeech and CMU Arctic bdl).

