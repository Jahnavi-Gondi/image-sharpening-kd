import os
from PIL import Image, ImageFilter

#config
base_path = "path to subset"
scaling_factor = 1.5
interpolation = Image.BICUBIC
gaussian_blur_radius = 0.8

input_folder = base_path  
output_folder = os.path.join(base_path, "blur_1.5x_gaussian_0.8")  

def apply_blur(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            img = Image.open(input_path).convert("RGB")
            width, height = img.size

            # Downscale
            downscaled = img.resize(
                (int(width / scaling_factor), int(height / scaling_factor)),
                resample=interpolation,
            )

            # Apply Gaussian Blur
            blurred = downscaled.filter(ImageFilter.GaussianBlur(radius=gaussian_blur_radius))

            # Upscale back to original size
            final_img = blurred.resize((width, height), resample=interpolation)

            # Save output PNG
            output_path = os.path.join(output_folder, filename)
            final_img.save(output_path, format="PNG")

    print(f"Done processing images in: {output_folder}")

# Run the function
apply_blur(input_folder, output_folder)
