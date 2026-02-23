import timm
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms

model_name = "vit_base_patch14_dinov2.lvd142m"
model = timm.create_model(model_name, pretrained=True, num_classes=0)
model.eval()

data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config)


def get_features(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        features = model(img_tensor)
    return features.squeeze().numpy()


try:
    feat1 = get_features("auto2.jpg")
    feat2 = get_features("auto1.jpg")

    similarity = cosine_similarity([feat1], [feat2])[0][0]

    print(f"--- RESULTS ---")
    print(f"Model: {model_name}")
    print(f"Car sim: {similarity:.4f}")

    if similarity > 0.85:
        print("Probably the same car.")
    else:
        print(".")

except FileNotFoundError:
    print("Error locating the images")
