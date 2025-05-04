# download_publaynet_fix.py
import urllib.request
import os

print("Downloading fresh model weights (this may take a while)...")
weights_url = "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"

# Make sure we're getting a fresh copy
if os.path.exists("model_final.pth"):
    os.rename("model_final.pth", "model_final.pth.old")

# Download the file
urllib.request.urlretrieve(weights_url, "model_final.pth")
print("Weights downloaded successfully!")
