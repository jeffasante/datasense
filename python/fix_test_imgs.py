from PIL import Image
import os

os.makedirs("test_dataset/images", exist_ok=True)
# Create a 256x256 blue image
img = Image.new('RGB', (256, 256), color = (73, 109, 137))
img.save('test_dataset/images/img1.png')
# Create a 128x128 green image
img = Image.new('RGB', (128, 128), color = (109, 73, 137))
img.save('test_dataset/images/img2.jpg')
print("Successfully created real test images.")
