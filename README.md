# Automatic Classification of Mechanical Components of Engines

This project leverages deep learning techniques to automatically classify mechanical components such as bearings, nuts, gears, and bolts. Accurate classification is vital for applications like sorting, inspection, and quality control in manufacturing processes.


# [Live Demo](https://automatic-classification-of-mechanical-components-of-engines.streamlit.app/)

## Features

- **Deep Learning Models**: Implements models like ResNet-50 for high-accuracy classification.
- **Grad-CAM Visualization**: Utilizes Gradient-weighted Class Activation Mapping to visualize model focus areas.
- **Streamlit Interface**: Provides an interactive web application for users to upload images and view classification results.

## Installation

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/SREESAIARJUN/Automatic-Classification-of-Mechanical-Components-of-Engines.git
    cd Automatic-Classification-of-Mechanical-Components-of-Engines
    ```

2. **Set Up a Virtual Environment** (optional but recommended):

    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```

3. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Download the Pre-trained Model**:

    Ensure the `resnet50_gradcam_model.pth` file is in the project directory. If not, download it from the provided source or contact the repository maintainer.

## Usage

1. **Run the Streamlit Application**:

    ```bash
    streamlit run streamlit_app.py
    ```

2. **Upload an Image**:

    Access the local Streamlit web interface, upload an image of a mechanical component, and view the classification results along with Grad-CAM visualizations.

## Project Structure

- `streamlit_app.py`: Main application script for the Streamlit interface.
- `requirements.txt`: Lists Python dependencies.
- `resnet50_gradcam_model.pth`: Pre-trained ResNet-50 model with Grad-CAM.
- `.devcontainer/`: Configuration files for development containers.
- `.github/`: GitHub-specific configurations and workflows.

## Contributing

Contributions are welcome! Please fork the repository, create a new branch for your feature or bugfix, and submit a pull request for review.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## References

- [Automatic Classification of Mechanical Components of Engines Using Deep Learning Techniques](https://zenodo.org/record/8265646)
- [Component Parts of Internal Combustion Engines](https://en.wikipedia.org/wiki/Component_parts_of_internal_combustion_engines)
