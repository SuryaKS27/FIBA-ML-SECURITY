import json
import os
import shutil
from time import time
import config
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch import nn
from utils.dataloader import PostTensorTransform, get_dataloader
from utils.utils import progress_bar
from torchvision import datasets, transforms, models
import copy
from PIL import Image
import cupy as cp
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os


def get_model(opt):
    netC = None
    optimizerC = None
    schedulerC = None

   
    if opt.dataset == "ISIC2019":
        netC = models.resnet50(pretrained=True)
        netC.fc = nn.Linear(netC.fc.in_features, opt.num_classes)
        netC = netC.to(opt.device)

    # Optimizer
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4)

    # Scheduler
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)

    return netC, optimizerC, schedulerC

def Fourier_pattern(img_,target_img,beta,ratio):
    img_=cp.asarray(img_)
    target_img=cp.asarray(target_img)
    #  get the amplitude and phase spectrum of trigger image
    fft_trg_cp = cp.fft.fft2(target_img, axes=(-2, -1))  
    amp_target, pha_target = cp.abs(fft_trg_cp), cp.angle(fft_trg_cp) 
    amp_target_shift = cp.fft.fftshift(amp_target, axes=(-2, -1))
    #  get the amplitude and phase spectrum of source image
    fft_source_cp = cp.fft.fft2(img_, axes=(-2, -1))
    amp_source, pha_source = cp.abs(fft_source_cp), cp.angle(fft_source_cp)
    amp_source_shift = cp.fft.fftshift(amp_source, axes=(-2, -1))

    # swap the amplitude part of local image with target amplitude spectrum
    bs,c, h, w = img_.shape
    b = (np.floor(np.amin((h, w)) * beta)).astype(int)  
    # 中心点
    c_h = cp.floor(h / 2.0).astype(int)
    c_w = cp.floor(w / 2.0).astype(int)

    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    amp_source_shift[:,:, h1:h2, w1:w2] = amp_source_shift[:,:, h1:h2, w1:w2] * (1 - ratio) + (amp_target_shift[:,:,h1:h2, w1:w2]) * ratio
    # IFFT
    amp_source_shift = cp.fft.ifftshift(amp_source_shift, axes=(-2, -1))

    # get transformed image via inverse fft
    fft_local_ = amp_source_shift * cp.exp(1j * pha_source)
    local_in_trg = cp.fft.ifft2(fft_local_, axes=(-2, -1))
    local_in_trg = cp.real(local_in_trg)

    return cp.asnumpy(local_in_trg)

def create_bd(inputs, opt, save_dir=None):
    """
    Create backdoor images.

    Args:
        inputs (torch.Tensor): Input images.
        opt: Options with hyperparameters.
        save_dir (str, optional): Directory to save backdoor images. Defaults to None.

    Returns:
        torch.Tensor: Backdoor images.
    """
    bs, _, _, _ = inputs.shape

    transforms_list = []
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    transforms_list.append(transforms.ToTensor())  
    transforms_class = transforms.Compose(transforms_list)

    im_target = Image.open(opt.target_img).convert('RGB')
    im_target = transforms_class(im_target)

    im_target = np.clip(im_target.numpy() * 255, 0, 255)
    im_target = torch.from_numpy(im_target).repeat(bs, 1, 1, 1)

    inputs = np.clip(inputs.numpy() * 255, 0, 255)

    bd_inputs = Fourier_pattern(inputs, im_target, opt.beta, opt.alpha)
    bd_inputs = torch.tensor(np.clip(bd_inputs / 255, 0, 1), dtype=torch.float32)

    # Save images if save_dir is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for i in range(bs):
            img = bd_inputs[i].permute(1, 2, 0).numpy() * 255
            img = img.astype(np.uint8)
            img_pil = Image.fromarray(img)
            img_pil.save(os.path.join(save_dir, f"bd_image_{i}.png"))

    return bd_inputs.to(opt.device)


def create_cross(inputs, opt):
    bs, _, _, _ = inputs.shape
    transforms_list = []
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    transforms_list.append(transforms.ToTensor())  
    transforms_class = transforms.Compose(transforms_list)

    ims_noise = []
    noiseImage_list = os.listdir(opt.cross_dir)
    noiseImage_names = np.random.choice(noiseImage_list,bs)
    for noiseImage_name in noiseImage_names:
        noiseImage_path = os.path.join(opt.cross_dir,noiseImage_name)

        im_noise = Image.open(noiseImage_path).convert('RGB')
        im_noise = transforms_class(im_noise)
        im_noise = np.clip(im_noise.numpy()*255,0,255)
        ims_noise.append(im_noise)

    inputs = np.clip(inputs.numpy()*255,0,255)
    ims_noise = np.array(ims_noise)
    cross_inputs = Fourier_pattern(inputs, ims_noise, opt.beta, opt.alpha)
    cross_inputs = torch.tensor(np.clip(cross_inputs/255,0,1),dtype=torch.float32)

    return cross_inputs.to(opt.device)

def eval(
    netC,
    test_dl,
    opt,
):
    print("Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_cross_correct = 0

    all_targets = []
    all_preds = []
    all_probs = []  # To store predicted probabilities for ROC curve

    save_bd_dir = "saved_bd_images"  # Directory for saving bd images

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean data
            preds_clean = netC(inputs)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)

            # Store clean targets and predictions for ROC curve
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(torch.argmax(preds_clean, 1).cpu().numpy())
            all_probs.extend(torch.softmax(preds_clean, dim=1).cpu().numpy())  # Get predicted probabilities

            # Evaluate Backdoor data
            inputs_bd = create_bd(copy.deepcopy(inputs.cpu()), opt, save_dir=save_bd_dir)
            if opt.attack_mode == "all2one":
                targets_bd = torch.ones_like(targets) * opt.target_label
            if opt.attack_mode == "all2all":
                targets_bd = torch.remainder(targets, opt.num_classes)
            preds_bd = netC(inputs_bd)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)

            acc_clean = total_clean_correct * 100.0 / total_sample
            acc_bd = total_bd_correct * 100.0 / total_sample

            # Evaluate cross
            if opt.cross_ratio:
                inputs_cross = create_cross(copy.deepcopy(inputs.cpu()), opt)
                preds_cross = netC(inputs_cross)
                total_cross_correct += torch.sum(torch.argmax(preds_cross, 1) == targets_bd)

                acc_cross = total_cross_correct * 100.0 / total_sample

                info_string = "BA: {:.4f} | ASR: {:.4f} | P-ASR: {:.4f}".format(acc_clean, acc_bd, acc_cross)
            else:
                info_string = "BA: {:.4f} - Best: {:.4f} | ASR: {:.4f} - Best: {:.4f}".format(
                    acc_clean, best_clean_acc, acc_bd, best_bd_acc
                )
            progress_bar(batch_idx, len(test_dl), info_string)

    # Compute ROC curve and AUC for each class (One-vs-Rest)
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Binarize the targets for multi-class ROC computation
    all_targets_bin = label_binarize(all_targets, classes=np.arange(opt.num_classes))

    plt.figure(figsize=(10, 8))

    # Compute ROC curve for each class
    for i in range(opt.num_classes):
        fpr, tpr, _ = roc_curve(all_targets_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label='Class {0} (AUC = {1:.2f})'.format(i, roc_auc))

    # Plot the diagonal line (chance level)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Finalize the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Each Class')
    plt.legend(loc='lower right')
    plt.show()

    # Confusion Matrix for Clean data
    cm = confusion_matrix(all_targets, all_preds)
    
    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=np.arange(opt.num_classes), yticklabels=np.arange(opt.num_classes))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

def main():
    # parameter prepare
    opt = config.get_arguments().parse_args()

    if opt.dataset == 'ISIC2019':
        opt.num_classes = 8
    else:
        raise Exception("Invalid Dataset")

    if opt.dataset == "ISIC2019":
        opt.input_height = 224
        opt.input_width = 224
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    # Dataset
    test_dl = get_dataloader(opt,False, set_ISIC2019='Test', pretensor_transform=False)
    # prepare model
    netC, optimizerC, schedulerC = get_model(opt)

    # Load pretrained model
    
    opt.ckpt_path = opt.test_model

    if os.path.exists(opt.ckpt_path):
        state_dict = torch.load(opt.ckpt_path)
        netC.load_state_dict(state_dict["netC"])
    else:
        print("Pretrained model doesnt exist")
        exit()
    eval(
        netC,
        test_dl,
        opt,
    )
if __name__ == "__main__":
    main()   