# Super Resolution for Images

---
## Project Description ⭐  
The application of `Generative Adversarial Networks`(GANs) in Computer Vision and Deep Learning has always fascinated me. Among the many real-world applications of GANs, `Image Inpainting` stands out, as it involves filling in missing or corrupted parts of an image using information from nearby areas. One of the most crucial appliations of image inpainting is `Super-resolution` which is the process of recovering and reconstructing the resolution of a noisy low-quality image into a very high-quality, high-resolution image. 

There are many state-of-the-art architectures that have been developed previously for the purpose of super-resolution. and `SRGAN`  is one of the earliest GANs that was developed for the super-resolution, and even though this performs very well, the main issue with this was it is highly compute intensive and slow. An improved version of the SRGAN called `SWIFT-SRGAN` was published in 2021. It focuses on improving the latency of the previous models for image super-resolution by reducing the computation size and introducing Depth-wise Separable Convolutions. This approach enables up-scaling low-resolution images into high-resolution images in real-time and with high efficiency, even on low-end computing devices, without compromising the clarity of the content.

The `Aim` of this project is to train and understand the working of SRGAN and Swift-SRGAN models on my proposed dataset. I will be downscaling high quality images from the dataset to generate low-resolution images (256x256). Then the generator will train to produce high-quality upscaled images (1024x1024), and these generated images will be compared against the original ground truths by the discriminator.  

&nbsp;  
**_Sample Results Comparison_** <table>
    <tr>
        <td><center><b>Low Resolution (LR) Input</b></center></td>
        <td><center><b>Super Resolution (SR) Output</b></center></td>
        <td><center><b>Ground Truth (HR) Original</b></center></td>
    </tr>
    <tr>
        <td>
            <center><img src="assets/sample_lr_input.png" height="250"></center>
        </td>
        <td>
            <center><img src="assets/sample_sr_output.png" height="250"></center>
        </td>
        <td>
            <center><img src="assets/sample_hr_input.png" height="250"></center>
        </td>
    </tr>
</table>


&nbsp;
## Data Sourcing & Processing   
For this project the Dataset used to train the Super Resolution model from Unsplash and pexels.
 
The original images have a high resolution of 1024 x 1024. To prepare this dataset for training a super resolution GAN, I downsampled the orignal high resolution images to 256 x 256 (one fourth) using BICUBIC interpolation from the PIL module. The downsampled images are served as an input to the generator architecture which then tries to generate a super resolution image which is as close as possible to the original higher resolution images. The data preprocessing script `prepare_data.py` is a part of the custom Train and Val data loader classes and is run automatically during the model training part. The data can be donwloaded using a script `make_dataset.py` and split into train and validation datasets using `split_dataset.py`. The scripts can be run as follows:

**Note**: Place your dataset in the appropriate directory on make_dataset.py.

**1. Create a new conda environment and activate it:** 
```
git clone https://github.com/alargam/Super_Resolution_Medical_X-Ray_Image.git
   cd Super_Resolution_Medical_X-Ray_Image-
```
**2. Install python package requirements:** 
```
pip install -r requirements.txt 
```
**3. Run the data downloading script:** 
```
python make_dataset.py
```
**4. Split the dataset into train and validation:** 
```
python split_dataset.py
```
This would create two files called `train_images.pkl` and `val_images.pkl` which would store the paths to train and validation split of images  


&nbsp;
## Deep Learning Model Architecture 🧨  
I have implemented the original [Swift-SRGAN](https://arxiv.org/pdf/2111.14320.pdf) model architecture to enhance the resolution of low-quality images. The authors trained the original Swift-SRGAN on [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset and the Flickr2K dataset. The Generative network was trained on a proposed dataset. Given an input image of size 256 x 256, the `Generator` generates a super-resolution image of size 1024 x 1024. The generated super resolution images are evaluated against the original high resolution images available in the dataset by the `Discriminator`.  

<img src="assets/network_architecture.png" height="400">


<br>  

**_Generator Architecture_**  
The generator consists of Depthwise Separable Convolutional layers, which are used to reduce the number of parameters and computation in the network. The major part of network is created of 16 residual blocks. Each block has a Depthwise Convolution followed by Batch Normalization, PReLU activation, another Depthwise Convolution, and Batch Normalizationa and finally a skip connection. After the residual blocks, the images is passsed thourgh upsample blocks and finally thourgh a convolution layer to generate the final output image.

<br>  
 
**_Discriminator Architecture_**  
The discriminator consists of 8 Depthwise Separable Convolutional blocks. Each block has a Depthwise Convolution followed by Batch Normalization and LeakyReLU activation. After the 8 blocks the images is passed through Avg Pooling and a fully connected layer to generate the final output. The objective of the discriminator is to classify super-resolution images generated by the geenrator as fake and original high-resolution images as real.  


&nbsp;
## Model Training and Evaluation 🚂  

The GAN model was evaluated and compared against ground truths using different metrics like Peak Signal to Noise Ratio (PSNR) and Structural Similarity Index (SSIM). 
- The PSNR is calculated by comparing the original signal to the processed signal, and it is expressed in decibels (dB). The higher the PSNR value, the less distortion or loss of information there is in the processed signal compared to the original signal.
- Similarly, SSIM lies between -1 and 1 and a higher SSIM score indicates a higher similarity between the two images structurally. 
- Compared to PSNR, SSIM is often considered a more perceptually accurate metric, as it takes into account the human visual system's sensitivity to changes in luminance, contrast, and structure.

The model was trained on T4 In Colab with a batch size of 2. Following are the metrics obtained after training the models on full dataset for 7 epochs:  

            
| Metric                              |       7 Epochs (DL)       | 
| ----------------------------------- | :-----------------------: |
| Peak Signal to Noise Ratio (PSNR)   |         36.1 (dB)         | 
| Structural Similarity Index (SSIM)  |            0.81           |
  

&nbsp;
### Following are the steps to run the model training code:

**1. To train the model using python script**
- You can train a model direcltly by runnning the driver python script : `train_model.py`
- You can pass `batch_size`, `num_epochs`, `upscale_factor` as arguments
- You will need a GPU to train the model
```
python train_model.py  --upscale_factor 4 --num_epochs 10 --batch_size 2
```
**5. Model checkpoints and results** 
- The trained genertor and Discriminator are saved to `/model/` directory after every epoch. The save format is `netG_{UPSCALE_FACTOR}x_epoch{epoch}.pth.tar`
- The metrics results are saved a csv to the `/logs/` folder with the filename `metrics_{epoch}_train_results.csv`  
  

&nbsp;
## Custom Loss Function 🎯  
I have used the same loss function mentioned by the authors of the Swift-SRGAN or SRGAN paper. The loss function for the Generator is a combination of multiple losses, each weighted and added together. The most crucial loss is the Perceptual Loss which is a combination of Adversarial Loss and Content Loss.
&nbsp;  
```
Total_Loss = Image_Loss + Perception_Loss + TV_Loss
```
```
Perceptual_Loss = Adversarial_Loss + Content_Loss
```  

&nbsp;  
**_Loss 1: Image Loss_**  
This is a naive loss functionn whihc calculates the Mean Squared Error b/w the generated image and the original high res image pixels.  
  
&nbsp;  
**_Loss 2: Content Loss_**  
It represents the information that is lost or distorted during the processing of an image. The image generated by the generator and the original high res image are passed though the MobileNetV2 network to compute the feature vectors of both the images. Content loss is calculated as the euclidean distance b/w the feature vectors of the original image and the generated image.  
  
&nbsp;  
**_Loss 3:  Adversarial Loss_**  
It is used to train the generator network by providing a signal on how to improve the quality of the generated images. This is calculated based on the discriminator's output, which indicates how well it can distinguish between the real and fake images. Generator tries to minimize this loss, by trying to generate images that the discriminator cannot distinguish.  
  
&nbsp;  
**_Loss 4:  Total Variation loss_**  
It measures the variation or changes in intensity or color between adjacent pixels in an image. It is defined as the sum of the absolute differences between neighboring pixels in both the horizontal and vertical directions in an image.  
  

&nbsp;  
## Project Structure 🧬  
The project data and codes are arranged in the following manner:

```
├── assets/                        # images and visual assets
├── data/                          # dataset (train_images.pkl, val_images.pkl)
├── src/                       # scripts for training, preprocessing, and evaluation
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
```  
