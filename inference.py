import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms

from src import UNet

model = UNet(3, 1)
ckpt = torch.load("checkpoints/epoch=23-train_loss=0.0419-val_loss=0.0396.ckpt")
model.load_state_dict(ckpt["state_dict"], strict=False)
model.eval()

input_image = Image.open("data/new_rotate/images/figure_1_0.png").convert("RGB")
transformers = transforms.Compose(
    [
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
input_image = transformers(img=input_image)
with torch.no_grad():
    output_mask = model(input_image.unsqueeze(0))

output_mask = output_mask.detach().cpu().squeeze(0).permute(1, 2, 0)
output_mask_np = output_mask[:, :, -1].numpy()
output_mask_np[output_mask_np > 0.5] = 255
output_mask_np[output_mask_np <= 0.5] = 0

print(output_mask)
Image.fromarray(output_mask_np).convert("L").save("test.png", "PNG", bits=1)
