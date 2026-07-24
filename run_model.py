# # import torch
# # import torch.nn as nn
# # from torchvision import models, transforms
# # from PIL import Image
# # import os

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # model_path = "ResNet50_plant_classifier.pth"
# # model_name = "ResNet50"
# # num_classes = 7

# # class_names = ["Apple", "Corn", "Grape", "Peach", "Potato", "Strawberry", "Tomato"]
# # # strawberry predicted correctly

# # image_path = "corn.jpeg"
# # img = Image.open(image_path).convert('RGB')

# # transform = transforms.Compose([
# #     transforms.Resize((224, 224)),
# #     transforms.ToTensor(),
# #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
# #                          std=[0.229, 0.224, 0.225])
# # ])

# # input_tensor = transform(img).unsqueeze(0).to(device)

# # # Load the model architecture + weights
# # model = models.resnet50(weights=None)
# # model.fc = nn.Linear(model.fc.in_features, num_classes)
# # model.load_state_dict(torch.load(model_path, map_location=device))

# # model.to(device)
# # model.eval()

# # with torch.no_grad():
# #     output = model(input_tensor)
# #     _, predicted = torch.max(output, 1)
# #     predicted_class = class_names[predicted.item()]

# # print(f"\nModel: {model_name}")
# # print(f"Input Image: {os.path.basename(image_path)}")
# # print(f"Predicted Class: {predicted_class}")



# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image
# import os

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model_path = "ResNet50_plant_classifier.pth"
# model_name = "ResNet50"
# num_classes = 7

# class_names = ["Apple", "Corn", "Grape", "Peach", "Potato", "Strawberry", "Tomato"]
# # strawberry predicted correctly

# image_path = "tomato_leaf.jpeg"
# img = Image.open(image_path).convert('RGB')

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# input_tensor = transform(img).unsqueeze(0).to(device)

# # Load the model architecture + weights
# model = models.resnet50(weights=None)
# model.fc = nn.Linear(model.fc.in_features, num_classes)
# model.load_state_dict(torch.load(model_path, map_location=device))

# model.to(device)
# model.eval()

# with torch.no_grad():
#     output = model(input_tensor)
#     _, predicted = torch.max(output, 1)
#     predicted_class = class_names[predicted.item()]

# print(f"\nModel: {model_name}")
# print(f"Input Image: {os.path.basename(image_path)}")
# print(f"Predicted Class: {predicted_class}")

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model_path = "VGGNet_plant_classifier.pth" for VGGNet
# model_path = "AlexNet_plant_classifier.pth" for AlexNet
model_path = "ResNet50_Plant_Classifier.pth"
model_name = "ResNet50"
num_classes = 7

class_names = ["Apple", "Corn", "Grape", "Peach", "Potato", "Strawberry", "Tomato"]
# strawberry predicted correctly

image_path = "apple_leaf.jpeg"
img = Image.open(image_path).convert('RGB')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

input_tensor = transform(img).unsqueeze(0).to(device)

# for AlexNet and VGGNet replace next line with:
# model = models.vgg16(weights=None) for VGGNet
# model = models.alexnet(weights=None) for AlexNet
model = models.resnet50(weights=None)

# for AlexNet and VGGNet replace next line with 
# model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes) 
model.fc = nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load(model_path, map_location=device))

model.to(device)
model.eval()

with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    predicted_class = class_names[predicted.item()]

print(f"\nModel: {model_name}")
print(f"Input Image: {os.path.basename(image_path)}")
print(f"Predicted Class: {predicted_class}")



# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image
# import os

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model_path = "VGGNet_plant_classifier.pth"
# model_name = "VGGNet"
# num_classes = 7

# class_names = ["Apple", "Corn", "Grape", "Peach", "Potato", "Strawberry", "Tomato"]
# # strawberry predicted correctly

# image_path = "apple_leaf.jpeg"
# img = Image.open(image_path).convert('RGB')

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# input_tensor = transform(img).unsqueeze(0).to(device)

# # Load the model architecture + weights
# model = models.vgg16(weights=None)
# model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
# model.load_state_dict(torch.load(model_path, map_location=device))

# model.to(device)
# model.eval()

# with torch.no_grad():
#     output = model(input_tensor)
#     _, predicted = torch.max(output, 1)
#     predicted_class = class_names[predicted.item()]

# print(f"\nModel: {model_name}")
# print(f"Input Image: {os.path.basename(image_path)}")
# print(f"Predicted Class: {predicted_class}")