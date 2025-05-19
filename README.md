# Chest X-Ray Classification Project

This project implements multiple deep learning models for chest X-ray classification, achieving high accuracy across different architectures. The models are deployed using FastAPI and containerized with Docker for easy deployment and usage.

## Models Performance

The project implements three different models, all achieving excellent test accuracy:

1. **DINOv2** - 98% test accuracy (BEST MODEL)
2. **EfficientNetB3 (Keras)** - 97.5% test accuracy
3. **Custom CNN** - 97.4% test accuracy 

## Project Structure

```
├── models/
│   ├── dinov2_model.py
│   ├── efficientnet_model.py
│   └── custom_cnn.py
├── api/
│   └── main.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup and Installation

1. Clone the repository:
```bash
git clone <https://github.com/Haaaaaaaaaax/Chest-X-Ray-Classification-Dockerization-and-Deployment.git>
cd chest-xray-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Models

### Local Development

1. Start the FastAPI server:
```bash
uvicorn api.main:app --reload
```

2. Access the API documentation at `http://localhost:8000/docs`

### Using Docker
1. Build the Docker image:
```bash
docker build -t chest-xray-classifier .
```

2. Run the container:
```bash
docker run -p 8000:8000 -gpus all chest-xray-classifier
```

## API Usage

The FastAPI server provides three endpoints for X-ray image classification, one for each model:

1. **DINOv2 Model (Best Performance)**
   - **Endpoint**: `/predict/dinov2`
   - **Method**: POST
   - **Input**: Form data with an image file
   - **Response**: JSON with classification results

2. **EfficientNetB3 Model**
   - **Endpoint**: `/predict/efficientnet`
   - **Method**: POST
   - **Input**: Form data with an image file
   - **Response**: JSON with classification results

3. **Custom CNN Model**
   - **Endpoint**: `/predict/cnn`
   - **Method**: POST
   - **Input**: Form data with an image file
   - **Response**: JSON with classification results

Example using curl for DINOv2 model:
```bash
curl -X POST "http://localhost:8000/predict/dinov2" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path_to_your_xray.jpg"
```

## Model Details
### DINOv2
- Based on the state-of-the-art DINOv2 architecture
- Achieves 98% test accuracy
- Leverages self-supervised learning capabilities

### EfficientNetB3
- Implemented using Keras
- Achieves 97% test accuracy
- Efficient architecture with good balance of accuracy and computational efficiency

### Custom CNN
- Built from scratch
- Achieves 98% test accuracy
- Optimized for chest X-ray classification

## Contributing

Feel free to submit issues and enhancement requests!

## License

MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
