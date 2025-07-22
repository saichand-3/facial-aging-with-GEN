Facial Aging Using SAM is a deep learning-based project that simulates realistic facial aging effects using StyleGAN2 and pSp (Pixel2Style2Pixel) encoding, optionally enhanced by SAM (Segment Anything Model) for precise facial segmentation. This tool allows users to transform facial images to older age versions while maintaining identity and photorealism.

✨ Features
🧓 Age Transformation: Apply realistic aging effects to facial images.

🧠 StyleGAN2 Integration: Leverages pretrained StyleGAN2 models for high-quality generation.

🔄 pSp Encoder: Converts real images into latent vectors suitable for manipulation.

🧩 Modular Design: Clean and modular codebase for easy customization.

🖼️ Sample Outputs: Includes sample images and similarity graphs to visualize effectiveness.

📁 Project Structure
bash
Copy
Edit
Facial_Aging_Using_SAM/
├── run_aging.py                # Main script to run the aging pipeline
├── config/
│   └── paths_config.py         # Path configuration for models and datasets
├── datasets/
│   └── augmentations.py        # Data augmentation utilities
├── models/
│   ├── psp.py                  # Pixel2Style2Pixel encoder
│   ├── encoders/               # Custom and pretrained encoders
│   └── stylegan2/              # Full StyleGAN2 implementation with CUDA ops
├── notebooks/
│   └── imageX.jpg              # Sample input images
├── results/
│   └── aging_similarity_graph1.png # Output similarity graph
🔧 Requirements
Python 3.8+

PyTorch

NVIDIA GPU with CUDA

Other dependencies listed in requirements.txt (you may need to create this)

🚀 Usage
bash
Copy
Edit
python run_aging.py --input /path/to/image.jpg --output /path/to/save
Make sure to update paths in config/paths_config.py before running.

📊 Results

🧠 Credits
StyleGAN2 (NVIDIA)

pSp Encoder (Pixel2Style2Pixel)

SAM (Segment Anything) (if integrated)

📄 License
MIT License. Feel free to use, modify, and distribute.

