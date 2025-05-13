# 🎨 AI-Powered Artistic Image Transformation

This project implements a **Neural Style Transfer model** using **Adaptive Instance Normalization (AdaIN)**. It trains a decoder on-the-fly using a content image and a style image, then generates a stylized output image. The entire workflow is packaged in a **Docker container** — no need for a pre-trained model!

## 🖼️ Example

Given:
- `input.jpg`: A real-world photo
- `style_transfer.jpg`: An artistic image (e.g., Van Gogh's painting)

The model generates:
- `output.jpg`: A stylized version of the input with the texture and tone of the style.

## 🗂 Project Structure


## 🚀 Running with Docker

### 1. Clone or download this repository.

git clone git@github.com:NabeelShabbir/CycleGAN.git
cd CycleGAN

docker build -t style-transfer-app .
docker run --rm -v ${PWD}/output:/app style-transfer-app

