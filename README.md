# Chest X-ray Classification Project

A full-stack machine learning application for classifying chest X-ray images using deep learning models.

## Project Structure

```
├── backend/          # Flask API server
│   ├── app.py       # Main Flask application
│   ├── requirements.txt
│   └── best_models/ # Trained model files
└── frontend/        # React frontend application
    ├── src/
    └── package.json
```

## Features

- Deep learning models for chest X-ray classification
- RESTful API backend built with Flask
- Modern React frontend with Tailwind CSS
- Support for multiple model architectures (ResNet18, ViT, VGG19)

## Setup

### Backend

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the Flask server:
   ```bash
   python app.py
   ```

### Frontend

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

## Technologies Used

- **Backend**: Flask, PyTorch, torchvision
- **Frontend**: React, Tailwind CSS
- **ML Models**: ResNet18, Vision Transformer (ViT), VGG19

## License

This project is for educational purposes.
