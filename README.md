# 🖼️ Image Shape Classifier

This project is a Streamlit application that allows you to upload a CSV file containing shape data, detects the shapes, and replaces them with regularized shapes (like circles, squares, rectangles, lines, etc.). The shapes are then saved as images, and the application uses the Replicate API to classify the shapes.

## Features

- **Shape Detection**: Detects various shapes from the provided CSV file, including circles, squares, rectangles, and lines.
- **Image Processing**: Saves detected shapes as images with a white background.
- **Shape Classification**: Uses the Replicate API to classify the shapes into categories such as square/rectangle, circle, ellipse, triangle, star, or none of the above.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/image-shape-classifier.git
   cd ADOBE-CURVETOPIA
   ```


## Dependencies Installation

### Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Set up the `.env` file:

Create a `.env` file in the root directory of the project and add your Replicate API key:

```
REPLICATE_API_TOKEN = "your_api_key"
```

### Upload a CSV File:

* Once the app is running, you'll be prompted to upload a CSV file.
* The CSV file should contain shape data in a specific format.

### Shape Detection and Saving:

* The app processes the CSV file, detects the shapes, and saves them as images in the `output_images` directory.

### Shape Classification:

* The saved images are then classified using the Replicate API, and the results are displayed on the Streamlit interface.
