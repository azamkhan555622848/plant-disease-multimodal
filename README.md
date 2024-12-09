# Plant Disease Analysis with CLIP and Multimodal Fusion

This repository implements a multimodal approach to plant disease analysis using CLIP (Contrastive Language-Image Pre-training) to combine visual features from plant leaf images with semantic textual data. The system can perform disease classification and severity estimation.

## Features

- CLIP-based multimodal fusion of images and text
- Disease severity estimation using combined embeddings
- Support for multiple plant diseases
- Interpretable severity levels (Low, Moderate, High)
- Extensible architecture for adding new diseases

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/plant-disease-multimodal.git
cd plant-disease-multimodal
```

2. Create a Python environment (Python 3.8+ required):
```bash
conda create -n multimodal_env python=3.8
conda activate multimodal_env
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
project_root/
├─ data/
│  ├─ images/            # Raw plant leaf images
│  ├─ images_enhanced/   # ESRGAN-enhanced images
│  ├─ agrold/            # Textual data and descriptions
│  └─ samples/           # Example samples
├─ models/
│  ├─ clip/             # CLIP model integration
│  └─ checkpoints/      # Saved model checkpoints
├─ src/
│  ├─ models/
│  │  ├─ clip_model.py
│  │  └─ severity_estimator.py
│  ├─ utils/
│  │  ├─ download_sample_images.py
│  │  ├─ prepare_images.py
│  │  └─ create_disease_descriptions.py
│  ├─ extract_embeddings.py
│  ├─ fuse_embeddings.py
│  ├─ train_severity_estimator.py
│  └─ predict_severity.py
├─ requirements.txt
└─ README.md
```

## Usage

### 1. Prepare Data

Download and prepare the plant disease images:
```bash
python src/utils/download_sample_images.py
python src/utils/prepare_images.py
```

### 2. Extract Embeddings

Extract CLIP embeddings from images and text:
```bash
# For images
python src/extract_embeddings.py --mode image --input_path data/images_enhanced --output_path embeddings/images_embeddings.pt

# For text
python src/extract_embeddings.py --mode text --input_path data/agrold/descriptions.txt --output_path embeddings/text_embeddings.pt
```

### 3. Fuse Embeddings

Combine image and text embeddings:
```bash
python src/fuse_embeddings.py \
  --image_embeddings embeddings/images_embeddings.pt \
  --text_embeddings embeddings/text_embeddings.pt \
  --output embeddings/multimodal_embeddings.pt
```

### 4. Train Severity Estimator

Train the severity estimation model:
```bash
python src/train_severity_estimator.py \
  --embeddings embeddings/multimodal_embeddings.pt \
  --output models/severity_estimator \
  --epochs 50
```

### 5. Predict Disease Severity

Use the trained model to predict severity for new images:
```bash
python src/predict_severity.py \
  --model models/severity_estimator/best_model.pt \
  --image path/to/your/image.jpg \
  --text "Description of the disease symptoms" \
  --input_dim 1024
```

## Model Details

The implementation uses:
1. CLIP (Contrastive Language-Image Pre-training) for multimodal embeddings
2. Custom severity estimator with configurable architecture
3. Multiple fusion strategies (concatenation, addition, multiplication)

## Dataset

The project uses:
- Plant Village dataset for images
- Custom disease descriptions following AgroLD format
- ESRGAN-enhanced images for better feature extraction

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for the CLIP model
- Plant Village dataset creators
- AgroLD for the disease information format