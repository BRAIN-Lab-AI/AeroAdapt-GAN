# AeroAdapt-GAN


# Adversarial Unsupervised Domain Adaptation for Aerial Image Segmentation

## Introduction
The proposed Enhanced CycleGAN-DeeplabV3 model is a cutting-edge deep learning approach designed for unsupervised domain adaptation in semantic segmentation of aerial imagery, leveraging advanced generative adversarial techniques to bridge significant domain shifts between source and target datasets. By integrating a powerful CycleGAN architecture with a robust DeeplabV3 segmentation network, complemented by ResNet50 as the backbone, custom mean-IoU-based early stopping criteria, and strategic regularization techniques such as dropout and learning rate decay, the model efficiently translates images across different sensor modalities, iteratively refining segmentation outputs. This sophisticated integration not only addresses the persistent challenges of domain-induced performance degradation and training instability seen in traditional adaptation methods but also significantly enhances generalization, consistency, and accuracy, thereby opening new possibilities for reliable and scalable remote sensing applications in urban planning, environmental monitoring, and geospatial intelligence.

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
- **ResNet50 Backbone:** A deep convolutional neural network featuring residual connections designed to mitigate vanishing gradient issues and enhance feature extraction.
- **Mean Intersection-over-Union (Mean IoU):** A metric measuring segmentation accuracy by quantifying the overlap between predicted and actual segmented regions across multiple classes.
- **Early Stopping:** A regularization technique halting model training once validation performance metrics cease to improve, thus preventing overfitting.
- **Adversarial Loss:** Loss used in GAN training to measure how well the generator network fools the discriminator network.
- **Unsupervised Domain Adaptation:** A learning scenario adapting a model trained on labeled source-domain data to unlabeled target-domain data without requiring additional labeling.
- **Domain Shift:** Variations in data distributions between source and target domains caused by differences in sensors, resolutions, illumination, or environmental conditions.
- **Source Domain:** The dataset on which the model is initially trained, containing labeled examples.
- **Target Domain:** The dataset to which the trained model is applied, usually differing in characteristics from the source domain and typically unlabeled.

### Problem Statements
- **Problem 1:** Semantic segmentation models exhibit significant performance degradation when trained on aerial images from one domain (e.g., RGB images from Potsdam) and tested on another domain (e.g., IRRG images from Vaihingen) due to substantial domain shifts.
- **Problem 2:** Existing unsupervised domain adaptation methods frequently struggle to maintain semantic consistency and structural integrity during image translation, resulting in unrealistic translated images and reduced segmentation accuracy.
- **Problem 3:** Current semantic segmentation models inadequately handle discrepancies arising from sensor variations, differences in spatial resolution, and diverse class representations in aerial imagery, leading to limited model generalization and effectiveness.
  
### Research Areas
- **Advanced Domain Adaptation Methods:** Developing novel architectures and techniques to handle complex domain shifts involving diverse sensor modalities and varying spatial resolutions in aerial imagery.
- **Efficient Training and Optimization:** Investigating lightweight models, more efficient training algorithms, and resource-aware approaches to facilitate deployment in resource-constrained environments.
- **Semi-Supervised and Weakly Supervised Approaches:** Extending the framework to scenarios with limited labeled data, leveraging semi-supervised and weakly supervised learning techniques to reduce labeling effort while maintaining performance.

  
### Loopholes
- **Domain Shift Sensitivity:** Models remain sensitive to significant domain shifts caused by large variations in sensor modalities, resolutions, and class distributions.
- **Translation Quality:** Potential inconsistencies in maintaining structural and semantic integrity during unsupervised image translation, affecting downstream segmentation tasks.
- **Computational Efficiency:** The CycleGAN and DeepLabV3-ResNet50 integration demands substantial computational resources (e.g., high-performance GPUs), limiting practical deployment and scalability.

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Adaptive Domain Alignment:**
Develop adaptive alignment mechanisms within the architecture to dynamically reduce domain shifts by better aligning source and target domain features.
2. **Semantic-Aware Loss Functions:**
Integrate advanced semantic-aware loss functions (e.g., cycle-consistency combined with semantic segmentation loss) to ensure high-quality translation while preserving semantic structures.
3. **Efficient Computational Strategies:**
Implement lightweight neural network components and optimized training techniques (e.g., mixed-precision training, structured pruning) to significantly decrease computational resource demands while maintaining segmentation accuracy.

### Proposed Solution: Code-Based Implementation
This repository provides an implementation of an enhanced unsupervised domain adaptation model for semantic segmentation using TensorFlow and Keras. The solution includes:

- **DeepLabV3-ResNet50 Segmentation Model:** Implements a robust semantic segmentation framework optimized for multi-scale feature extraction through atrous spatial pyramid pooling (ASPP).
- **CycleGAN Domain Adaptation:** Utilizes dual generator-discriminator pairs with dropout and instance normalization layers to perform effective unsupervised image translation between source and target domains.
- **Custom Early Stopping:** Incorporates a mean Intersection-over-Union (mean IoU) metric for monitoring validation performance, ensuring optimal training efficiency and preventing overfitting.
- **Automated Data Preprocessing Pipeline:** Systematically partitions large aerial imagery into uniformly sized patches, automating data handling and ensuring consistent, repeatable preprocessing steps.
- **Comprehensive Evaluation Metrics:** Provides detailed segmentation evaluation using accuracy, precision, recall, F1-score, confusion matrices, and mean IoU to thoroughly assess model performance.

### Key Components
- **`model.py`**: Contains the modified UNet architecture and other model components.
- **`train.py`**: Script to handle the training process with configurable parameters.
- **`utils.py`**: Utility functions for data processing, augmentation, and metric evaluations.
- **`inference.py`**: Script for generating images using the trained model.

## Model Workflow
The workflow of the proposed CycleGAN-DeeplabV3 model is designed to perform unsupervised domain adaptation for semantic segmentation of aerial imagery through an integrated translation and segmentation process:

1. **Input:**
   - **Dataset Preparation:** Large aerial images from the source domain (Potsdam, RGB) and target domain (Vaihingen, IRRG) are systematically partitioned into 512×512 pixel tiles to create structured training and validation datasets.
   - **Initial Segmentation Labels:** Source domain image patches are paired with their corresponding semantic segmentation masks to form training sets.

2. **Domain Adaptation Process:**
   - **CycleGAN Image Translation:** A CycleGAN architecture, consisting of dual generator-discriminator networks, translates images from the source domain to visually match characteristics of the target domain without requiring paired examples.
   - **Iterative Refinement:** The CycleGAN employs cycle-consistency loss to iteratively refine generated images, preserving structural and semantic integrity while minimizing domain-induced discrepancies.

3. **Segmentation Fine-Tuning:**
   - **Translated Image Dataset:** Translated images are automatically paired with original segmentation masks, forming a new training dataset that closely aligns with target domain conditions.
   - **DeepLabV3 Model Training:** The segmentation model, featuring a ResNet50 backbone and ASPP module, undergoes fine-tuning on this translated dataset, utilizing early stopping guided by a custom mean Intersection-over-Union (mean IoU) metric to achieve optimal segmentation accuracy and prevent overfitting.
     
4. **Output:**
   - **Segmentation Predictions:** The fine-tuned model produces pixel-level semantic segmentation masks on target domain images, accurately identifying classes such as buildings, vegetation, and roads, effectively overcoming the domain shift.
   - **Quantitative and Qualitative Evaluation:** Outputs are evaluated using detailed metrics (accuracy, precision, recall, F1-score, mean IoU) and visually assessed via confusion matrices and per-class segmentation comparisons to confirm performance improvements.

## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/BRAIN-Lab-AI/AeroAdapt-GAN.git
    cd AeroAdapt-GAN

    ```

2. **Set Up the Environment:**
    Create a virtual environment and install the required dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Run the Notebook:**
    Open the Notebook and follow the cells to train and test the models:
    ```bash
    jupyter notebook AeroAdapt-GAN.ipynb
    ```
4. **Export or Convert to Script (Optional):**
    If needed, notebook can be converted to a Python script:
    ```bash
    jupyter nbconvert --to script AeroAdapt-GAN.ipynb
    python AeroAdapt-GAN.py
    ```

4. **Generate Results:**
    After training, check the outputs generated in the notebook. Adjust input settings or parameters in the cells if you want to try different conditions (e.g., for CycleGAN translation or segmentation evaluation)
    Visual results (segmentation maps, translated images) are automatically saved to:
    ```bash
    ./results/
    ```

## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of TensorFlow, Keras, OpenCV, and related open-source libraries for their exceptional resources and continuous development efforts.
- **Individuals:** Special thanks to our supervisor Dr. Muzammil Behzad for invaluable guidance and support throughout this project.
- **Resource Providers:** Gratitude to to the International Society for Photogrammetry and Remote Sensing (ISPRS) for providing comprehensive aerial imagery datasets that enabled thorough experimentation.
- **Research Community:** Appreciation to researchers whose foundational work and insights in unsupervised domain adaptation and semantic segmentation significantly guided this research.


