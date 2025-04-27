# AeroAdapt-GAN


# Adversarial Unsupervised Domain Adaptation for Aerial Image Segmentation

## Introduction
The proposed Enhanced CycleGAN-DeeplabV3 model is a cutting-edge deep learning approach designed for unsupervised domain adaptation in semantic segmentation of aerial imagery, leveraging advanced generative adversarial techniques to bridge significant domain shifts between source and target datasets. By integrating a powerful CycleGAN architecture with a robust DeeplabV3 segmentation network, complemented by ResNet101 as the backbone, custom mean-IoU-based early stopping criteria, and strategic regularization techniques such as dropout and learning rate decay, the model efficiently translates images across different sensor modalities, iteratively refining segmentation outputs. This sophisticated integration not only addresses the persistent challenges of domain-induced performance degradation and training instability seen in traditional adaptation methods but also significantly enhances generalization, consistency, and accuracy, thereby opening new possibilities for reliable and scalable remote sensing applications in urban planning, environmental monitoring, and geospatial intelligence.

## Project Metadata
### Authors
- **Team:** Maha Balhareth, Mushyirah Alharbi and Razan Alhumud
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** KFUPM

### Project Documents
- **Presentation:** [Project Presentation](/presentation.pptx)
- **Report:** [Project Report](/report.pdf)

### Reference Paper
- [Unsupervised Domain Adaptation Using Generative Adversarial Networks for Semantic Segmentation of Aerial Images](https://www.mdpi.com/2072-4292/11/11/1369)

### Reference Dataset
- [2D Semantic Labeling](https://www.isprs.org/education/benchmarks/UrbanSemLab/semantic-labeling.aspx)



## Project Technicalities

### Terminologies
- **Semantic Segmentation:** A pixel-level image classification process assigning class labels to each pixel in an image.
- **Generative Adversarial Networks (GANs):** Neural networks comprising two components (generator and discriminator) trained adversarially to produce realistic synthetic data.
- **CycleGAN:** A type of generative adversarial network utilizing cycle-consistency loss for translating images between unpaired source and target domains.
- **Adversarial Training:** Training methodology involving competition between generator and discriminator to achieve realistic synthetic data generation.
- **DeepLabV3 Architecture:** A state-of-the-art segmentation model employing atrous spatial pyramid pooling (ASPP) for capturing multi-scale contextual information.
- **ResNet101 Backbone:** A deep convolutional neural network featuring residual connections designed to mitigate vanishing gradient issues and enhance feature extraction.
- **Mean Intersection-over-Union (Mean IoU):** A metric measuring segmentation accuracy by quantifying the overlap between predicted and actual segmented regions across multiple classes.
- **Early Stopping:** A regularization technique halting model training once validation performance metrics cease to improve, thus preventing overfitting.
- **Adversarial Loss:** Loss used in GAN training to measure how well the generator network fools the discriminator network.
- **Unsupervised Domain Adaptation:** A learning scenario adapting a model trained on labeled source-domain data to unlabeled target-domain data without requiring additional labeling.
- **Domain Shift:** Variations in data distributions between source and target domains caused by differences in sensors, resolutions, illumination, or environmental conditions.
- **Source Domain:** The dataset on which the model is initially trained, containing labeled examples.
- **Target Domain:** The dataset to which the trained model is applied, usually differing in characteristics from the source domain and typically unlabeled.

### Problem Statements
- **Problem 1:** Achieving high-resolution and detailed images using conventional diffusion models remains challenging.
- **Problem 2:** Existing models suffer from slow inference times during the image generation process.
- **Problem 3:** There is limited capability in performing style transfer and generating diverse artistic variations.

### Loopholes or Research Areas
- **Evaluation Metrics:** Lack of robust metrics to effectively assess the quality of generated images.
- **Output Consistency:** Inconsistencies in output quality when scaling the model to higher resolutions.
- **Computational Resources:** Training requires significant GPU compute resources, which may not be readily accessible.

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Optimized Architecture:** Redesign the model architecture to improve efficiency and balance image quality with faster inference.
2. **Advanced Loss Functions:** Integrate novel loss functions (e.g., perceptual loss) to better capture artistic nuances and structural details.
3. **Enhanced Data Augmentation:** Implement sophisticated data augmentation strategies to improve the model’s robustness and reduce overfitting.

### Proposed Solution: Code-Based Implementation
This repository provides an implementation of the enhanced stable diffusion model using PyTorch. The solution includes:

- **Modified UNet Architecture:** Incorporates residual connections and efficient convolutional blocks.
- **Novel Loss Functions:** Combines Mean Squared Error (MSE) with perceptual loss to enhance feature learning.
- **Optimized Training Loop:** Reduces computational overhead while maintaining performance.

### Key Components
- **`model.py`**: Contains the modified UNet architecture and other model components.
- **`train.py`**: Script to handle the training process with configurable parameters.
- **`utils.py`**: Utility functions for data processing, augmentation, and metric evaluations.
- **`inference.py`**: Script for generating images using the trained model.

## Model Workflow
The workflow of the Enhanced Stable Diffusion model is designed to translate textual descriptions into high-quality artistic images through a multi-step diffusion process:

1. **Input:**
   - **Text Prompt:** The model takes a text prompt (e.g., "A surreal landscape with mountains and rivers") as the primary input.
   - **Tokenization:** The text prompt is tokenized and processed through a text encoder (such as a CLIP model) to obtain meaningful embeddings.
   - **Latent Noise:** A random latent noise vector is generated to initialize the diffusion process, which is then conditioned on the text embeddings.

2. **Diffusion Process:**
   - **Iterative Refinement:** The conditioned latent vector is fed into a modified UNet architecture. The model iteratively refines this vector by reversing a diffusion process, gradually reducing noise while preserving the text-conditioned features.
   - **Intermediate States:** At each step, intermediate latent representations are produced that increasingly capture the structure and details dictated by the text prompt.

3. **Output:**
   - **Decoding:** The final refined latent representation is passed through a decoder (often part of a Variational Autoencoder setup) to generate the final image.
   - **Generated Image:** The output is a synthesized image that visually represents the input text prompt, complete with artistic style and detail.

## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/enhanced-stable-diffusion.git
    cd enhanced-stable-diffusion
    ```

2. **Set Up the Environment:**
    Create a virtual environment and install the required dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Train the Model:**
    Configure the training parameters in the provided configuration file and run:
    ```bash
    python train.py --config configs/train_config.yaml
    ```

4. **Generate Images:**
    Once training is complete, use the inference script to generate images.
    ```bash
    python inference.py --checkpoint path/to/checkpoint.pt --input "A surreal landscape with mountains and rivers"
    ```

## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of PyTorch, Hugging Face, and other libraries for their amazing work.
- **Individuals:** Special thanks to bla, bla, bla for the amazing team effort, invaluable guidance and support throughout this project.
- **Resource Providers:** Gratitude to ABC-organization for providing the computational resources necessary for this project.


