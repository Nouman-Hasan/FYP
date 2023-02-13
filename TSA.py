from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import streamlit as st
import nibabel as nib
from tqdm.notebook import tqdm
from celluloid import Camera

import imgaug.augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt


from Dataset import CardiacDataset
from model import UNet
from Segmentation import AtriumSegmentation

from IPython.display import HTML
from plotly.graph_objs import *

torch.manual_seed(0)
model = AtriumSegmentation()
model = AtriumSegmentation.load_from_checkpoint(r"logs2/logs2/lightning_logs/version_0/checkpoints/epoch=48-step=11858.ckpt")

device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
model.eval();
model.to(device)

def prediction(subject):
    
    subject_mri = nib.load(str("imagesTs/"+str(subject.name))).get_fdata()
    def normalize(full_volume):
        mu = full_volume.mean()
        std = np.std(full_volume)
        normalized = (full_volume - mu) / std
        return normalized

    def standardize(normalized):
        standardized_data = (normalized - normalized.min()) / (normalized.max() - normalized.min())
        return standardized_data

    subject_mri = subject_mri[32:-32, 32:-32]
    standardized_scan = standardize(normalize(subject_mri))
    
    preds = []
    for i in range(standardized_scan.shape[-1]):
        slice = standardized_scan[:,:,i]
        with torch.no_grad():
            pred = model(torch.tensor(slice).unsqueeze(0).unsqueeze(0).float().to(device))[0][0]
            pred = pred > 0.5
        preds.append(pred.cpu())
        
    fig = plt.figure()
    camera = Camera(fig)

    for i in range(standardized_scan.shape[-1]):
        plt.imshow(standardized_scan[:,:,i], cmap = "bone")
        mask = np.ma.masked_where(preds[i] == 0, preds[i])
        plt.imshow(mask, alpha = 0.5)
        camera.snap()
    animation = camera.animate()

    
    return HTML(animation.to_html5_video())
    
    
# def main():
    
#     st.title("The Smart Atrium")
    
#     subject = st.file_uploader(label = "Please choose a file")

#     # st.file_uploader("Choose a CSV file", accept_multiple_files=True)
#     # prediction(subject)
    
# if __name__ == "main":

st.title("The Smart Atrium")
st.markdown("Upload MRI")
subject = st.file_uploader(label = "Please choose a file")
if subject is not None:
    st.write(prediction(subject))
