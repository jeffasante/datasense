import os
from PIL import Image

paths = ["test_dataset/images/img1.png", "test_dataset/images/img2.jpg"]
for p in paths:
    abs_p = os.path.abspath(p)
    print(f"Checking: {abs_p} (Exists: {os.path.exists(abs_p)})")
    try:
        with Image.open(abs_p) as img:
            print(f"Resolution: {img.size}")
    except Exception as e:
        print(f"Error: {e}")
