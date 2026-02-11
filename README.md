# Super Resolution for Medical Images ☠️

Super-Resolution for Medical Images 🩻

Archit | Spring '23 | Duke AIPI 540 Final Project

Project Overview ⭐

Generative Adversarial Networks (GANs) have revolutionized image processing tasks like super-resolution and image inpainting. Super-resolution aims to reconstruct high-resolution images from low-resolution inputs, a process especially critical in medical imaging (X-rays, CT scans, MRIs) where high-resolution images can improve diagnosis but capturing them often requires higher radiation doses.

Instead of exposing patients to more radiation, we can capture low-resolution images (~256×256) and enhance them using GAN-based super-resolution to produce high-quality outputs (~1024×1024). This project explores training SRGAN and Swift-SRGAN architectures on chest X-ray images to improve image clarity while maintaining computational efficiency.

Dataset & Preprocessing 💾

Dataset: NIH Chest X-rays
 – 112,120 high-resolution images (1024×1024) from 30,805 patients.

Preprocessing:

High-res images downsampled to 256×256 using bicubic interpolation.

Train/validation split generated with split_dataset.py.

Scripts for dataset preparation:

python ./scripts/make_dataset.py       # download dataset from Kaggle
python ./scripts/split_dataset.py      # split into train/val sets

Baseline: Non-Deep Learning Approach ✨

Before using GANs, a simple bicubic interpolation was applied to upsample low-res images. Limitations include:

Cannot capture complex textures or fine details.

Poor generalization to unseen data.

Run baseline:

python ./scripts/non_dl_super_resolution.py


Metrics are saved to: ./logs/non_dl_approach_metrics.csv

GAN Architecture 🧨
Generator

Uses Depthwise Separable Convolutions for efficiency.

16 residual blocks with skip connections, BatchNorm, and PReLU activations.

Upsample blocks generate the high-resolution image (1024×1024).

Discriminator

8 depthwise separable convolution blocks with LeakyReLU activations.

Ends with average pooling + fully connected layer to classify images as real/fake.

Loss Functions 🎯

Total Loss:

Total_Loss = Image_Loss + Perceptual_Loss + TV_Loss
Perceptual_Loss = Adversarial_Loss + Content_Loss


Image Loss: MSE between generated and original images.

Content Loss: Euclidean distance between MobileNetV2 feature vectors of generated and real images.

Adversarial Loss: Guides generator to produce realistic images by fooling the discriminator.

Total Variation Loss: Smooths pixel intensity variations for more natural images.

Training & Evaluation 🚂

Hardware: NVIDIA RTX6000, batch size = 16

Metrics: PSNR and SSIM
| Method | PSNR (dB) | SSIM |
|----------------------|------------|-------|
| Swift-SRGAN (10 epochs) | 41.66 | 0.96 |
| Swift-SRGAN (1 epoch) | 30.37 | 0.83 |
| Bicubic (baseline) | 30.40 | 0.74 |

Run training:

python ./scripts/train_model.py --upscale_factor 4 --num_epochs 10 --batch_size 16


Checkpoints saved to ./models/
Training metrics saved to ./logs/metrics_{epoch}_train_results.csv

Demo: Streamlit App 🧪

Run interactive demo:

git clone https://github.com/architkaila/Super-Resolution-for-Medical-Images
git checkout streamlit-demo
conda create -n st_demo python=3.9.16
conda activate st_demo
pip install -r requirements.txt
streamlit run streamlit_app.py


Play with the model here

Project Structure 🗂️
├── assets/                        # images and visual assets
├── data/                          # dataset (train_images.pkl, val_images.pkl)
├── src/                           # scripts for training, preprocessing, and evaluation
│   ├── custom_loss.py
│   ├── make_dataset.py
│   ├── model_architecture.py
│   ├── model_metrics.py
│   ├── non_dl_super_resolution.py
│   ├── prepare_data.py
│   ├── split_dataset.py
│   └── train_model.py
├── .gitignore
├── README.md
├── requirements.txt

Risks & Limitations ⚠️

Low-res images (<128×128) may lead to unrealistic details.

Generated images may produce non-existent features.

Ethical concerns: bias, accountability, and transparency.

Mitigation: diverse dataset + perceptual loss to maintain fidelity.

References 📚

NIH Chest X-rays Dataset

Ledig et al., SRGAN, 2017 – Paper

Krishnan et al., Swift-SRGAN, 2021 – Paper

SRGAN Loss Explanation

TensorFlow SRGAN Implementation
