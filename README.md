# Brain CT Scan Analysis Web Application

A Flask-based web application for analyzing Brain CT scans using deep learning and explainable AI. Users can register, log in, upload CT scan images, and receive predictions about potential abnormalities, along with explainable insights powered by a local LLM.

## Features

- **User Authentication:** Register and log in using an SQLite backend.
- **Image Upload & Processing:** Upload Brain CT scans; view processed images (grayscale, edge detection, thresholding, sharpening).
- **Automated Diagnosis:** Classifies scans using a pre-trained ResNet50 model into categories such as:
  - Different types of Hemorrhages
  - Skull Fractures
  - Different types of strokes
  - Sinusitis
- **Explainable AI:** Integrates with Ollama (gemma3:4b or medgemma3:4b model) to provide textual explanations for predictions.
- **Model Training:** Includes scripts and plots for model training and evaluation.

## Project Structure

- `app.py` — Main Flask web application.
- `RESNET_50_TRAIN.py` — Script for training the ResNet50 model.
- `ResNet50_model.h5` — Pre-trained model weights.
- `class_names.pkl` — Class labels for predictions.
- `explainable_ai.py` — Explainable AI integration with Ollama.
- `user_data.db` — SQLite database for user data.
- `accuracy_plot.png`, `loss_plot.png` — Model training plots.

## Requirements

- [uv](https://github.com/astral-sh/uv) (Python environment & package manager)
- [Ollama](https://ollama.com/) (for local LLM inference)
- Python 3.8+

## Setup & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/Sayed-Afnan-Khazi/FYP-BrainCTScanAnalysis.git
cd FYP-BrainCTScanAnalysis
```

### 2. Install Python Dependencies

```bash
uv sync
```

### 3. Start Ollama with the Required Model

Ensure [Ollama](https://ollama.com/download) is installed and running.

Ensure the `gemma3:4b` model is available. If not, you can pull it using:

```bash
ollama pull gemma3:4b
```

If you already have the model, you can start Ollama with:

```bash
ollama run gemma3:4b
```

### 4. Run the Web Application

```bash
uv run app.py
```

The app will be available at [http://localhost:5004](http://localhost:5004).

### 5. Using the Application

- Register for an account.
- Log in.
- Upload a Brain CT scan image.
- View processed images and model predictions.
- Get explainable AI insights for each prediction.

## Model Training

To retrain the model or update class names:

```bash
python RESNET_50_TRAIN.py
```

This will generate new model weights and update `accuracy_plot.png` and `loss_plot.png`.

## Notes

- Make sure `ResNet50_model.h5` and `class_names.pkl` are present in the project directory.
- Ollama must be running locally for explainable AI features to work.
- The application uses SQLite for user data storage; ensure `user_data.db` is accessible.
- For production use, consider using a more robust database and web server setup.
- The application is designed for educational purposes and may require further enhancements for production readiness, one of which is the insecure database connection and sql injection vulnerabilities. Use prepared statements or ORM for database interactions in production.