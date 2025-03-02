# Brain-Computer Interface Session Adaptation

This project investigates methods to improve the robustness of motor imagery-based Brain-Computer Interfaces (BCIs) across different recording sessions. We compare traditional Common Spatial Patterns (CSP) with adaptive approaches to address session-to-session variability in EEG signals.

## Project Overview

Brain-Computer Interfaces based on motor imagery enable communication and control through imagined movements. However, these systems face challenges with session-to-session variability due to factors like:
- Electrode placement shifts
- User fatigue
- Cognitive state changes
- Environmental conditions

This project implements and compares multiple approaches:
1. Classic CSP + LDA (baseline)
2. CORAL Adaptation
3. TA-CSPNN (Temporal-Adaptive CSP Neural Network)
4. Deep CORAL with custom training

## Data Structure

```
data/
├── raw/
│   ├── train/
│   │   ├── Competition_train_cnt.txt    # Raw EEG training data
│   │   └── Competition_train_lab.txt    # Training labels
│   └── test/
│       ├── test.txt                     # Raw EEG test data
│       └── test_label.txt               # Test labels
├── preprocessed/
│   ├── train/
│   │   ├── X_train_filt.npy            # Filtered training features
│   │   └── y_train.npy                  # Training labels
│   ├── validation/
│   │   ├── X_val_filt.npy              # Filtered validation features
│   │   └── y_val.npy                    # Validation labels
│   └── test/
│       ├── X_test_filt.npy             # Filtered test features
│       └── y_test.npy                   # Test labels
```

## Implementation Structure

```
.
├── main.py                 # Main script to run all pipelines
├── models/
│   ├── csp.py             # Classic CSP implementation
│   ├── coral.py           # CORAL domain adaptation
│   ├── deep_coral.py      # Deep CORAL network architecture
│   ├── evaluate.py        # Model evaluation utilities
│   ├── tacnn.py          # TA-CSPNN implementation
│   └── train.py          # Training pipelines
├── utils/
│   ├── preprocessing.py   # Data loading and preprocessing
│   └── metrics.py        # Performance metrics
└── outputs/
    ├── confusion_matrices/  # Saved confusion matrices
    ├── performance_plots/   # Performance visualization
    └── results.csv         # Evaluation metrics
```

## Setup and Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install numpy scipy sklearn tensorflow
```

## Usage

Run the main script to execute all pipelines:
```bash
python main.py
```

This will:
1. Load and preprocess the EEG data
2. Train and evaluate the baseline CSP+LDA model
3. Apply CORAL adaptation
4. Train and evaluate TA-CSPNN
5. Train Deep CORAL with domain adaptation

## Data Processing Pipeline

1. **Preprocessing**:
   - Bandpass filtering (8-30 Hz)
   - Trial segmentation
   - Train/validation split

2. **Feature Extraction**:
   - CSP spatial filtering
   - Log-variance feature computation
   - Multi-band decomposition (for TA-CSPNN)

3. **Domain Adaptation**:
   - CORAL feature alignment
   - Deep CORAL representation learning

## Results

Performance metrics and visualizations are saved in the `outputs/` directory:
- Confusion matrices
- Accuracy plots
- Cross-session performance comparisons

## References

- Common Spatial Patterns (CSP)
- CORAL Domain Adaptation
- Temporal-Adaptive CSP Neural Networks