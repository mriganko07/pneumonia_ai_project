### Project Title

**Pneumonia AI**

-----

### Project Description

This project provides two trained models, one for X-ray images and another for non-X-ray images, to help detect pneumonia. These models are saved in `.h5` and `.tflite` formats, making them suitable for deployment in applications like an Android app or a web app.

-----

### Installation

Follow these steps to set up the project locally:

**1. Create and Navigate to Project Directory**

```bash
mkdir pneumonia_ai
cd pneumonia_ai
```

**2. Set Up a Virtual Environment**

```bash
python -m venv venv
```

**3. Activate the Virtual Environment**

  - **On Windows:**

<!-- end list -->

```bash
venv\Scripts\activate
```

  - **On macOS and Linux:**

<!-- end list -->

```bash
source venv/bin/activate
```

**4. Install Dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**5. Verify TensorFlow Installation**

  - Check your TensorFlow version:

<!-- end list -->

```bash
python -c "import tensorflow as tf; print(tf._version_)"
```

  - If you have an NVIDIA GPU, you can install the CUDA-enabled version of TensorFlow for better performance:

<!-- end list -->

```bash
pip install tensorflow[and-cuda]
```

  - Verify that your GPU is recognized by TensorFlow:

<!-- end list -->

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

-----

### Usage

This section explains how to run the training scripts to generate the models.

**1. Generate Trained Models**

  - Run the `research_xray.py` script to train the model on X-ray images. This will generate the trained model files for X-ray detection.

<!-- end list -->

```bash
python research_xray.py
```

  - Run the `research_non_xray.py` script to train the model on non-X-ray images. This will generate the trained model files for non-X-ray detection.

<!-- end list -->

```bash
python research_non_xray.py
```

**2. Trained Model Files**
After running the scripts, the following files will be generated in your project directory:

  - `xray_preclassifier.h5` and `pneumonia_model.tflite`
  - `non_xray_model.h5` and `xray_preclassifier.tflite`

These `.h5` and `.tflite` files are the trained models ready to be integrated into an Android or web application.

----
