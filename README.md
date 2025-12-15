# Object Detection with YOLOS and Gradio

This project demonstrates an object detection application using the Hugging Face Transformers library with a pre-trained YOLOS model fine-tuned on the Fashionpedia dataset. It processes images to detect objects, draws bounding boxes with confidence scores, and provides a natural language summary of detections. A Gradio interface is included for easy interaction.

## Features
- Load images from URLs or upload via Gradio.
- Perform object detection using the `valentinafeve/yolos-fashionpedia` model.
- Render bounding boxes and labels on the detected objects.
- Generate a natural language summary of the predictions (e.g., "In this image, there are twenty-eight persons...").
- Suppress common warnings from the Transformers library.
- Interactive demo via Gradio for input/output visualization.

## Requirements
- Python 3.12+
- Libraries:
  - `transformers` (for the object detection pipeline)
  - `Pillow` (PIL for image handling)
  - `matplotlib` (for rendering bounding boxes)
  - `requests` (for loading images from URLs)
  - `inflect` (for natural language number-to-word conversion)
  - `gradio` (for the web interface)
  - `torch` (dependency for Transformers)

Install dependencies:
```
pip install transformers pillow matplotlib requests inflect gradio torch
```

**Note:** The project uses a specific model from Hugging Face. Ensure you have internet access for the first run to download the model weights.

## Files
- **helper.py**: Contains utility functions for loading images, rendering detection results, summarizing predictions in natural language, and ignoring warnings.
- **object_detection.ipynb**: Jupyter notebook that:
  - Imports utilities from `helper.py`.
  - Sets up the object detection pipeline.
  - Processes an example image.
  - Creates and launches a Gradio demo.
  - Demonstrates prediction summarization.

## Usage

### Running the Notebook
1. Open `object_detection.ipynb` in Jupyter Notebook or Jupyter Lab.
2. Run the cells sequentially:
   - Import utilities.
   - Load the object detection pipeline: `pipe = pipeline("object-detection", model="valentinafeve/yolos-fashionpedia")`.
   - Load an example image: `pil_image = load_image_from_url('http://images.cocodataset.org/val2017/000000039769.jpg')`.
   - Run detection: `pipline_output = pipe(pil_image)`.
   - Render results: `processed_image = render_results_in_image(pil_image, pipline_output)`.
   - Summarize: `summary = summarize_predictions_natural_language(pipline_output)`.
   - Launch Gradio demo: `demo.launch(share=True)` (this starts a local web server at `http://127.0.0.1:7860` and optionally shares a public link).
3. In the Gradio interface:
   - Upload an image or provide a URL.
   - The output will show the image with bounding boxes and labels.

**Note:** There's a typo in the notebook (`pipline` instead of `pipeline`). You may want to correct it for clarity.

### Example Output
- For the COCO dataset image (cats on a couch), the model might detect general objects like "person", "cell phone", etc., despite being fine-tuned on fashion. Adjust the model if needed for specific domains.
- Natural language summary example: "In this image, there are twenty-eight persons one cell phone two benchs two clocks and two potted plants."

### Function Details (from `helper.py`)
- `load_image_from_url(url)`: Fetches and opens an image from a URL using PIL.
- `render_results_in_image(in_pil_img, in_results)`: Uses Matplotlib to draw green bounding boxes and red text labels (object name and confidence score) on the image. Returns the modified PIL image.
- `summarize_predictions_natural_language(predictions)`: Counts detected objects and generates a sentence like "In this image, there are [count] [label]s...".
- `ignore_warnings()`: Suppresses specific warnings from Transformers and other libraries to clean up output.

## Limitations
- The model (`valentinafeve/yolos-fashionpedia`) is optimized for fashion items but can detect general objects. For better accuracy on non-fashion images, consider a different model like `hustvl/yolos-tiny`.
- No internet access in the code execution environment beyond initial model download.
- Gradio share links are temporary and may not work in restricted environments.
- Warning suppression is optional; call `ignore_warnings()` if needed.

## Contributing
Feel free to fork and improve! Suggestions: Add support for video input, batch processing, or custom models.

## License
MIT License. See [LICENSE](LICENSE) for details (if not present, assume open-source for educational purposes).
