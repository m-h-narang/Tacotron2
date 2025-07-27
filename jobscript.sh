#!/bin/bash
#SBATCH --job-name=tacotron2_inference
#SBATCH --output=/scratch/s6028608/DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/related_to_running_python_code/output.log
#SBATCH --error=/scratch/s6028608/DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/related_to_running_python_code/error.log
#SBATCH --gres=gpu:1

module --force purge
module --ignore_cache load Python/3.8.6-GCCcore-10.2.0
module --ignore_cache load CUDA/11.8.0
module --ignore_cache load cuDNN/8.7.0.84-CUDA-11.8.0

source /scratch/s6028608/venvs/first_env/bin/activate 

cd /scratch/s6028608/DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2

python inference.py --tacotron2 /scratch/s6028608/DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/tacotron2_1032590_6000_amp --waveglow /scratch/s6028608/DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/waveglow_1076430_14000_amp --wn-channels 256 -o related_to_running_python_code/output -i phrases/phrase.txt --fp16