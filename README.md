```yaml
library_name: transformers
language:
- fa
license: apache-2.0
base_model: openai/whisper-large-v3
tags:
- generated_from_trainer
datasets:
- mozilla-foundation-common-voice-17-0
metrics:
- wer
model-index:
- name: Whisper LargeV3 Persian - Persian ASR
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: common-voice-17-0
      type: mozilla-foundation-common-voice-17-0
      config: fa
      split: None
      args: 'config: Persian, split: train+validation+validated+test'
    metrics:
    - name: Wer
      type: wer
      value: 26.85
```
# 🎯 TL;DR
Fine-tuned Whisper Large V3 for Persian Automatic Speech Recognition (ASR)
📊 Trained on ~370K noisy Persian samples (Common Voice 17.0)
📉 Reduced WER from 34.6% → 26.85% after 7 epochs on NVIDIA H100

👉 [View Model on HuggingFace](https://huggingface.co/MohammadReza-Halakoo/1-persian-whisper-large-v3-train-validation-tested-100-percent-17-0-7-epoch)

# 🎙️ Whisper LargeV3 Persian - Persian ASR  

This is a fine-tuned version of [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)  
for **Persian Automatic Speech Recognition (ASR) or Speech-to-text** It was trained on the Common Voice 17.0 dataset and optimized for noisy environments.  


---

## ✨ Highlights
- 🗣️ Trained on **370K Persian audio samples** (40% noisy)  
- 📉 Reduced **WER** from **34.6% → 26.85%** after 7 epochs  
- ⚡ Supports **long audio transcription** with improved robustness in noisy environments  
- 🔬 Optimized with **LR tuning, noise augmentation, mixed-precision AMP**  

---

## 📌 Model Description
The Whisper Large-V3 Persian model is a fine-tuned version of OpenAI’s Whisper large model,  
specifically adapted for Persian ASR. It uses **Mel-spectrogram features** and advanced tokenization,  
providing strong transcription performance in real-world noisy conditions.  

---

## 🚀 Intended Uses & Limitations
**Intended Uses** ✅  
- Transcribing Persian audio into text  
- Voice assistants, transcription services, educational apps  
- Handling noisy real-world environments  

**Limitations** ⚠️  
- Lower accuracy on **rare dialects**  
- May require additional fine-tuning for **domain-specific audio**  

---

## 📊 Training & Evaluation Data
- Dataset: **Mozilla Common Voice 17.0 (Persian)**  
- Training samples: **368,172** (≈40% noisy)  
- Evaluation metric: **WER**  

---

## ⚙️ Training Procedure
**Hyperparameters:**  
- learning_rate: `1e-07`  
- train_batch_size: `20`  
- eval_batch_size: `20`  
- optimizer: `Adam (β1=0.9, β2=0.999, ε=1e-08)`  
- lr_scheduler: `linear (warmup=500 steps)`  
- num_epochs: `7`  
- precision: `AMP (fp16)`  

---

## 📈 Learning Rate Experiments
| Learning Rate | Epoch | Eval Loss | WER   |
|:-------------:|:-----:|:---------:|:-----:|
| 5e-06         | 3     | 0.4506    | 38.34% |
| 4.375e-06     | 3     | 0.3191    | 30.99% |
| 4e-06         | 3     | 0.2765    | 27.48% |
| 1e-05         | 3     | 0.1734    | 22.52% |
| 1e-06         | 3     | 0.1528    | 18.05% |
| 1e-07         | 3     | 0.1577    | 14.86% |
| 1e-08         | 3     | 0.1577    | 14.86% |

**Conclusion:**  
- ✅ Best learning rates: **1e-07, 1e-08** (stable gradients & lowest WER)  
- ❌ Higher LRs caused gradient instability and worse WER  

---

## 📉 Training Results
- Final training loss: **0.2004**  
- Final validation loss: **0.2368**  
- Final WER: **26.85%**  

### WER by Epoch
| Epoch | WER    | Change   |
|:-----:|:------:|:--------:|
| 1     | 34.60% | -        |
| 2     | 31.42% | -3.18%   |
| 3     | 29.37% | -2.05%   |
| 4     | 28.17% | -1.20%   |
| 5     | 27.42% | -0.75%   |
| 6     | 26.95% | -0.47%   |
| 7     | 26.85% | -0.10%   |

---

## 🖥️ Hardware Utilization
- GPUs: **2 × NVIDIA H100 PCIe (80GB)**  
- RAM: **17.05 GiB**  
- Swap: **1.9 GiB (≈100%)**  
- Training runtime: **10.6 days**  

---

## 🔮 Future Improvements
- More **epochs** (+8 → WER < 20%)  
- **Data augmentation** & noise filtering  
- **Dropout / regularization** for better generalization  
- Exploring **hybrid ASR architectures**  

---

📬 Contact

📧 Email: halakoo.mohammadreza@gmail.com

🤗 HuggingFace: [MohammadReza Halakoo](https://huggingface.co/MohammadReza-Halakoo)

💼 LinkedIn: [MohammadReza Halakoo](https://www.linkedin.com/in/mohammadreza-halakoo)

## 📦 Installation
```bash
pip install torch torchaudio transformers datasets accelerate jiwer librosa evaluate soundfile
