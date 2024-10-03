# 🌭 Hot Dog or Not Hot Dog: A Deep Learning Adventure in Image Classification

<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExamdjYndyOWZlbHFyOTc2MWtqaWlkeGxzcGZqeGR3MWRsZmkydzVheSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/l0Iy9iqThC2ueLTkA/giphy.gif" alt="Hot Dog Not Hot Dog GIF" width="100%">

## Problem Statement

In the ever-evolving landscape of artificial intelligence and computer vision, the ability to accurately classify images remains a cornerstone challenge. This project tackles a seemingly whimsical yet surprisingly complex task: distinguishing hot dogs from non-hot dogs in images. Inspired by the fictional "SeeFood" app from the TV show Silicon Valley, we embark on a journey to create a robust, real-world implementation of this concept.

## Key Questions

1. How effectively can deep learning models distinguish hot dogs from other food items and objects?
2. Which convolutional neural network (CNN) architectures perform best for this binary classification task?
3. Can ensemble methods or fine-tuning techniques significantly improve classification accuracy?
4. How does the model's performance translate to a real-time, user-friendly application?

## Overview

This project leverages state-of-the-art deep learning techniques to build and compare various CNN models for hot dog classification. We explore transfer learning, ensemble methods, and fine-tuning to push the boundaries of classification accuracy. The culmination of this effort is a Streamlit web application that allows users to upload images and receive instant predictions.

## Data Dictionary

| Feature | Description |
|---------|-------------|
| Image   | 224x224 pixel RGB image |
| Label   | Binary: 0 (Hot Dog), 1 (Not Hot Dog) |

**Note:** The original dataset structure inverted the typical binary classification convention, with 0 representing the positive class (Hot Dog).

## Dataset Structure

```
data/
├── hotdog-nothotdog/
    ├── test/
    │   ├── hotdog/
    │   └── nothotdog/
    └── train/
        ├── hotdog/
        └── nothotdog/
```

- **Training** images: 2,121 hot dogs, 2,121 non-hot dogs
- **Test** images: 200 hot dogs, 200 non-hot dogs
- Total images: 4,642

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Streamlit
- Plotly

## Executive Summary

Our hot dog classification project achieved remarkable results, with our best model (fine-tuned InceptionV3) reaching an accuracy of 95.25% on the test set. We compared multiple CNN architectures, including InceptionV3, DenseNet121, and MobileNetV2, and explored ensemble methods to further boost performance. The project culminated in a user-friendly Streamlit application that demonstrates the model's capabilities in real-time.

## Insights and Findings

1. **Architecture Performance**: InceptionV3 consistently outperformed other architectures, likely due to its depth and inception modules that capture features at multiple scales.

2. **Transfer Learning Efficacy**: Utilizing pre-trained weights significantly accelerated training and improved performance, highlighting the power of transfer learning in specialized image classification tasks.

3. **Fine-tuning Benefits**: Fine-tuning the best-performing model (InceptionV3) yielded a small but notable improvement, increasing accuracy from 95.00% to 95.25%.

4. **Ensemble Limitations**: Surprisingly, our ensemble method (95.25% accuracy) did not outperform the fine-tuned InceptionV3 model, suggesting that the individual models might be making similar errors.

5. **Class Balance Impact**: The balanced nature of our dataset (equal hot dog and not hot dog images) contributed to the high performance across both classes, as evidenced by the confusion matrices.

6. **Model Ranking**:
   - InceptionV3 (Fine-tuned): 95.25%
   - InceptionV3: 95.00%
   - Ensemble: 93.75%
   - DenseNet121: 91.50%
   - MobileNetV2: 88.50%

## Model Architecture

Our champion model utilizes the InceptionV3 architecture, pre-trained on ImageNet and fine-tuned on our hot dog dataset. Key features include:

- Input Shape: (299, 299, 3)
- Global Average Pooling
- Dense layer with 1024 units and ReLU activation
- Output layer with sigmoid activation for binary classification

## Implementation Steps

1. **Data Preparation**: 
   - Use `ImageDataGenerator` for efficient data loading and augmentation.
   - Apply transformations like rotation, shifting, shearing, and zooming to increase dataset diversity.

2. **Model Creation and Training**:
   - Implement InceptionV3, DenseNet121, and MobileNetV2 with transfer learning.
   - Train models using early stopping and learning rate reduction callbacks.

3. **Evaluation and Comparison**:
   - Use accuracy, confusion matrices, and classification reports for assessment.
   - Visualize training progress and compare model performances.

4. **Advanced Techniques**:
   - Create an ensemble of the trained models.
   - Fine-tune the best-performing model (InceptionV3).

5. **Streamlit App Development**:
   - Create a user-friendly interface for image upload and classification.
   - Implement the logic flip to correctly interpret model predictions.

## Streamlit Demo

Our Streamlit application provides an intuitive interface for users to interact with the hot dog classification model. Due to the unique labeling in our dataset (0 for hot dog, 1 for not hot dog), we implemented a logic flip in the app to correctly interpret the model's output.

### Logic Flip Implementation

```python
if raw_confidence is not None:
    # Flip the confidence score
    confidence = 1 - raw_confidence

    st.write("## Results")
    if confidence > 0.5:
        st.success(f"It's a hot dog! 🌭 (Confidence: {confidence:.2%})")
    else:
        st.error(f"It's not a hot dog! ❌ (Confidence: {(1-confidence):.2%})")
```

## Tips for Improvement

1. Start with a basic network architecture and gradually add complexity.
2. Use image augmentation and transfer learning to enhance model performance.
3. Save your best model for use in the Streamlit app.
4. Focus on building the Streamlit app early in the process to identify integration issues.
5. Utilize GPU acceleration for faster training of CNNs.

## Limitations and Future Work

1. **Dataset Specificity**: Expand the dataset to include a wider variety of hot dogs and non-hot dogs.
2. **Image Quality Dependency**: Improve robustness to various image qualities and conditions.
3. **Binary Nature**: Explore multi-class classification for different types of hot dogs or other food items.
4. **Computational Efficiency**: Investigate model compression techniques for more efficient deployment.

## Conclusion

The "Hot Dog or Not Hot Dog" classifier demonstrates the power of modern deep learning techniques in solving specific image classification tasks. Through careful model selection, training, and optimization, we've created a highly accurate classifier that can distinguish hot dogs from other images with over 95% accuracy. The Streamlit demo showcases how such models can be deployed in user-friendly applications, bridging the gap between advanced AI research and practical, everyday use.

This project not only serves as a testament to the capabilities of transfer learning and fine-tuning in computer vision but also highlights the importance of understanding and adapting to unique dataset characteristics, as evidenced by our need to flip the classification logic in the final application.

<img src="https://i.redd.it/y583w8qasg121.jpg" alt="Hot Dog Not Hot Dog Meme" width="100%">

## Acknowledgments

- Original concept inspired by the TV show Silicon Valley
- Dataset provided by [The Data Sith on Kaggle](https://www.kaggle.com/datasets/thedatasith/hotdog-nothotdog)
- Project structure and initial readme by Greg (Chuck) Dye and Jeff Hale

Happy hot dog hunting! 🌭🔍

