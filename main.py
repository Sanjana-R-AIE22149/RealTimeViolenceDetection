import torch
import os
import cv2
from transformer_model import ViolenceTransformerModel
from mongo_handler import MongoDBLogger
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model =ViolenceTransformerModel().to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_video(folder_path):
    images = []
    for img_file in sorted(os.listdir(folder_path))[:10]:
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(img).to(device)
        images.append(img)

   
    images = torch.stack(images).unsqueeze(0).to(device)  # (1, T, C, H, W)

    
    T = images.shape[1]
    D = 128
    motion_vectors = torch.zeros((1, T, D)).to(device)

    with torch.no_grad():
        pred = model(images, motion_vectors)
        pred = (pred + 1) / 2

        print(pred.shape)  

       
        if pred > 0.5:
            predicted_class = 1  
        else:
            predicted_class = 0  

    return predicted_class


# logger = MongoDBLogger()
# base_path = "ExtractedFrames"

# for label in ["Violence", "NonViolence"]:
#     folder = os.path.join(base_path, label)
#     for video in os.listdir(folder):
#         video_path = os.path.join(folder, video)
#         pred = predict_video(video_path)
#         logger.insert_prediction(video, "Violence" if pred == 1 else "NonViolence")

pred=predict_video(r"C:\Users\sanjana\OneDrive\Desktop\Big Data Project\ExtractedFrames\Violence\V_8")
if pred==1:
    print("Violence")
else:
    print("Non-Violence")