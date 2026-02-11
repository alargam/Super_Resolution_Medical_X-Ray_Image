import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
import math
import pandas as pd
import pickle
from tqdm import tqdm
from model_metrics import ssim
from torchvision import transforms

def naive_super_resolution(image_path, lr_scale=256, hr_scale=1024):
    hr_image = Image.open(image_path).convert('RGB')
    lr_image = transforms.Resize((lr_scale, lr_scale), interpolation=Image.BICUBIC)(hr_image)
    hr_restore_img = transforms.Resize((hr_scale, hr_scale), interpolation=Image.BICUBIC)(lr_image)
    return to_tensor(lr_image), to_tensor(hr_restore_img), to_tensor(hr_image)

def run_pipeline(val_data_list, batch_size=1):
    results = {"mse":0, "ssims":0, "psnr":0, "ssim":0, "batch_sizes":0}
    val_bar = tqdm(val_data_list, total=len(val_data_list))
    for image_path in val_bar:
        results["batch_sizes"] += batch_size
        lr_img, hr_restore, hr_img = naive_super_resolution(image_path)
        batch_mse = ((hr_restore - hr_img) ** 2).mean()
        results["mse"] += batch_mse * batch_size
        batch_ssim = ssim(hr_restore.unsqueeze(0), hr_img.unsqueeze(0)).item()
        results["ssims"] += batch_ssim * batch_size
        results["psnr"] = 10*math.log10((hr_img.max()**2)/(results["mse"]/results["batch_sizes"]))
        results["ssim"] = results["ssims"]/results["batch_sizes"]
        val_bar.set_description(f"PSNR: {results['psnr']:.4f} dB SSIM: {results['ssim']:.4f}")
    return results

if __name__ == "__main__":
    with open('../data/val_images.pkl','rb') as f:
        val_data_list = pickle.load(f)
    results = run_pipeline(val_data_list)
    print(f"PSNR: {results['psnr']:.4f} dB SSIM: {results['ssim']:.4f}")
    pd.DataFrame([results]).to_csv("../logs/non_dl_approach_metrics.csv", index=False)
