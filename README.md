# FIBA-ML-SECURITY

# Adversarial Attacks and Robust Defenses for Machine Learning Models

This research explores **Frequency Injection-based Backdoor Attacks (FIBA)** on machine learning models and evaluates **Spectral Defense** as a mitigation strategy. The project focuses on vulnerabilities in the frequency domain of images and proposes a detection mechanism to enhance model robustness.

## Key Components
1. **FIBA (Attack)**:  
   - Embeds backdoor triggers in the frequency domain using Fourier transforms.  
   - Preserves visual semantics while manipulating model predictions.  
   - Achieves high Attack Success Rate (ASR: 99.6%) with minimal perturbation (p-ASR: 7.73%).  

2. **Spectral Defense (Countermeasure)**:  
   - Detects adversarial samples via Fourier domain analysis.  
   - Logistic Regression classifier achieves **100% accuracy** (AUC: 1.0).  
   - Integrated as a preprocessing filter to block poisoned inputs.  

## Datasets
- **ISIC-2019**: Medical imaging dataset (25,331 dermoscopic images across 8 classes).  
- **Evaluation**: 3-fold cross-validation.  

## Implementation
### FIBA Attack
- **Model**: ResNet50 (Adam optimizer, LR=0.01, 200 epochs).  
- **Parameters**: Blend ratio (`α=0.15`), frequency patch (`β=0.10`).  
- **Metrics**:  
  - Benign Accuracy (BA): 84.09% (vs. 86.15% clean).  
  - Sensitivity: 84.10%, Specificity: 98.30%.  

### Spectral Defense
- **Classifier**: Logistic Regression.  
- **Metrics**:  
  - Precision: 1.00, Recall: 0.71 (Class 1), F1-score: 0.92 (macro).  
  - Confusion Matrix: 5072 TN (Class 0), 5 TP (Class 1).  

## Results
- **FIBA Effectiveness**:  
  - High ASR with low perceptible distortion (see Figure 9 for examples).  
  - Clean accuracy drops marginally (2.06% relative).  
- **Spectral Defense**:  
  - Perfect ROC-AUC (1.00) and 100% benign sample detection.  
  - Integrated pipeline reduces ASR to near-zero while maintaining BA.  

## Figures & Tables
- **Figure 1**: FIBA framework.  
- **Figure 2**: Spectral Defense workflow.  
- **Figure 3**: Integrated pipeline.  
- **Table I**: Comparative metrics (Clean vs. Attacked model).  
- **Figures 5-8**: Accuracy trends, confusion matrix, ROC curves.  
- **Figures 10-12**: Spectral Defense evaluation metrics.  
