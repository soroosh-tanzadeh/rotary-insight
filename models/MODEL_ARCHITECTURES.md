# Model Architectures Documentation

This document provides a comprehensive overview of all neural network architectures implemented in the `models` directory for time-series classification tasks.

---

## Table of Contents

1. [DSICNN](#1-dsicnn)
2. [ResNet1D](#2-resnet1d)
3. [DenseNet1D](#3-densenet1d)
4. [EfficientNet1D](#4-efficientnet1d)
5. [WRN1D](#5-wrn1d)
6. [CNN_BiLSTM](#6-cnn_bilstm)
7. [CNN_SelfAttention](#7-cnn_selfattention)
8. [DPCCNN](#8-dpccnn)
9. [ResNetClassifier](#9-resnetclassifier)
10. [TransformerEncoderClassifier](#10-transformerencoderclassifier)
11. [Supporting Modules](#11-supporting-modules)

---

## 1. DSICNN

**File:** `DSICNN.py`

DSICNN (Dilated Separable Inverted Convolution Neural Network) is a lightweight CNN architecture designed for efficient feature extraction from 1D time-series data. It utilizes depthwise separable convolutions with dilation to capture patterns at different scales while maintaining a low parameter count. The core component is the DCIM (Dilated Convolution Inverted Module), which combines dilated depthwise convolution, pointwise convolution, and batch normalization.

---

## 2. ResNet1D

**File:** `Resnet1D.py`

A 1D adaptation of the classic ResNet architecture. It features residual blocks with skip connections, enabling the training of deeper networks by mitigating the vanishing gradient problem. The architecture consists of an initial convolutional block followed by three layers of stacked residual blocks with increasing channel depth, ending with a global average pooling and fully connected classifier.

---

## 3. DenseNet1D

**File:** `Densnet.py`

A 1D adaptation of DenseNet (Densely Connected Convolutional Networks). In this architecture, each layer receives feature maps from all preceding layers within a dense block, promoting strong feature reuse and gradient flow. The network comprises multiple dense blocks separated by transition layers that reduce spatial dimensions and channel counts.

---

## 4. EfficientNet1D

**File:** `EfficientNet.py`

A 1D adaptation of EfficientNet, utilizing Mobile Inverted Bottleneck Convolution (MBConv) blocks. These blocks use channel expansion, depthwise convolution, and projection to efficiently learn features. The architecture is designed to balance accuracy and efficiency, progressively increasing the number of channels through a series of MBConv blocks.

---

## 5. WRN1D

**File:** `WideResidualnet.py`

A Wide Residual Network (WRN) adapted for 1D signals. Unlike standard ResNets that increase depth, WRN focuses on increasing the width (number of channels) of the network. This approach allows for shallower but wider networks that can capture more features per layer. It uses wide residual blocks with a configurable widening factor.

---

## 6. CNN_BiLSTM

**File:** `CNN_BiLSTM.py`

A hybrid architecture that combines Convolutional Neural Networks (CNN) with Bidirectional Long Short-Term Memory (BiLSTM) networks. The CNN layers acts as a feature extractor to capture local spatial patterns in the signal, while the BiLSTM layers process the sequence of extracted features to capture temporal dependencies in both forward and backward directions.

---

## 7. CNN_SelfAttention

**File:** `CNN_selfattention.py`

A hybrid architecture combining CNN layers with a Self-Attention mechanism. The model first extracts local features using standard 1D convolutional layers. These features are then processed by a self-attention module, which calculates attention scores to weigh the importance of different parts of the signal, enabling the model to capture global context and long-range dependencies.

---

## 8. DPCCNN

**File:** `DPC_CNN.py`

DPCCNN is an advanced CNN architecture featuring Dilated Pointwise Convolutions with Squeeze-and-Excitation (SE) blocks and Global Attention Blocks (GAB). It integrates multiple mechanisms: dilated convolutions for extended receptive fields, SE blocks for channel-wise attention, and GAB for spatial attention, making it capable of learning complex patterns.

---

## 9. ResNetClassifier

**File:** `resnet.py`
**Class Name:** `ResNetClassfier` (Note: internal spelling is `Classfier`)

This model takes a unique approach by converting 1D time-series signals into 2D spectrograms using the Short-Time Fourier Transform (STFT). The resulting 2D images (magnitude spectrograms) are then processed by a standard 2D ResNet architecture with Bottleneck blocks. This allows the model to leverage powerful 2D vision techniques on time-series data.

---

## 10. TransformerEncoderClassifier

**File:** `transformer.py`

A pure Transformer architecture for time-series classification. It preprocesses the input signal (optionally using FFT) and effectively treats it as a sequence of patches. These patches are embedded and processed by a stack of Transformer Encoder blocks using multi-head self-attention. A learnable class token is used to aggregate information for the final classification.

---

## 11. Supporting Modules

### PositionalEncoder
**File:** `positional_encoder.py`
Adds sinusoidal positional encodings to token embeddings, allowing the Transformer to be aware of the sequence order.

### EmbeddingLinear
**File:** `embedding.py`
Handles the conversion of raw time-series signals into patch-based embeddings. It splits the signal into patches, projects them linearly, and optionally adds positional encodings and a class token.
