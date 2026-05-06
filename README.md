# MDM Method for Separation of Multiple Finite Sets

## Overview

This project implements the Generalized MDM method for separating multiple finite sets in Euclidean space. The algorithm solves the problem of projecting the origin onto the Minkowski sum of convex hulls of finite sets. The implementation includes both cyclic and extreme computational schemes and applies the method to multi-class image classification tasks.

## Mathematical Background

Given `s ≥ 2` finite sets `P_k = {A_{k1}, A_{k2}, ..., A_{kd_k}}` in `ℝⁿ`, where each `A_{kj} ∈ ℝⁿ` and `d_k ≥ 1`, we consider the Minkowski sum of convex hulls:
C = C₁ + C₂ + ... + C_s = {v₁ + ... + v_s | v_k ∈ conv(P_k)}


The goal is to solve the projection problem:
minimize (1/2)‖v‖² subject to v ∈ C

### Optimality Criterion

For `λ ∈ Λ` and `v = Aλ`, define:
Δ_k(λ) = max_{j ∈ M_k⁺(λ)} ⟨A_{kj}, v⟩ - min_{j} ⟨A_{kj}, v⟩
Δ(λ) = max_{k=1:s} Δ_k(λ)


**Theorem:** `v = Aλ` solves the problem ⇔ `Δ(λ) = 0`

## Computational Schemes

### Scheme 1: Extreme Choice
On each iteration, selects the set with maximum `Δ_k(λ)` and performs a correction step.

### Scheme 2: Cyclic Choice
Processes sets sequentially in order, performing correction steps when `Δ_k(λ) > 0`.

## Implementation

### Core Classes

#### `MDMSeparationSolver`
Core MDM method implementation.

**Parameters:**
- `scheme`: 'cyclic' or 'extreme' (default: 'cyclic')
- `tol`: Tolerance for optimality criterion (default: 1e-8)
- `max_iter`: Maximum number of iterations (default: 50000)

**Methods:**
- `fit(P_list)`: Solves the separation problem and returns `v_star` and separation distance

#### `ImprovedMDMMultiClassifier`
Multi-class classifier using One-vs-Rest strategy.

**Parameters:**
- `use_pca`: Apply PCA dimensionality reduction (default: False)
- `pca_components`: Number of PCA components (default: 32)
- `use_scaling`: Apply RobustScaler (default: True)

**Methods:**
- `train(features_dict, ...)`: Train classifier on extracted features
- `predict(image_path, return_details=False)`: Classify a single image
- `save_model(filepath)`: Save trained model
- `load_model(filepath)`: Load pre-trained model

### Feature Extraction

The `extract_enhanced_features_single` function extracts a comprehensive feature vector (35494 dimensions) including:

| Feature Type | Description |
|--------------|-------------|
| Color | HSV histograms (16 bins each), color moments (mean, std, skewness) |
| Texture | LBP (Local Binary Patterns), GLCM features (contrast, dissimilarity, homogeneity, energy, correlation) |
| Gradient | HOG (Histogram of Oriented Gradients) with 4×4 cells |
| Local Features | SIFT with statistical aggregation (mean, std, max, min) |
| Shape | Hu moments, circularity, area ratio |
| Frequency | DCT low-frequency components (16×16) |

## Experimental Results

### Synthetic Data (2000 experiments)

| Configuration | Scheme | Avg Iterations | Median | Time (sec) |
|---------------|--------|----------------|--------|-------------|
| n=100, s=30, d_k=350 | cyclic | 349.2 | 349 | 22.8 |
| n=100, s=30, d_k=350 | extreme | 10470 | 10470 | 49.7 |
| n=200, s=20, d_k=450 | cyclic | 449.8 | 449 | 29.4 |
| n=200, s=20, d_k=450 | extreme | 8980 | 8980 | 63.1 |

**Key findings:**
- Cyclic scheme requires **25–300 times fewer iterations** than extreme scheme
- Cyclic scheme shows more stable and predictable convergence
- Extreme scheme exhibits heavy-tailed distribution with occasional very slow convergence

### Image Classification Results

#### Two-class model (roses vs dandelions)

| Class | Accuracy | Precision | Recall | F1-score |
|-------|----------|-----------|--------|----------|
| roses | 73.87% | 0.742 | 0.739 | 0.740 |
| dandelions | 74.12% | 0.741 | 0.743 | 0.742 |
| **Average** | **74.03%** | **0.741** | **0.741** | **0.741** |

#### Three-class model (roses vs dandelions vs sunflowers)

| Class | Accuracy | Precision | Recall | F1-score |
|-------|----------|-----------|--------|----------|
| roses | 54.69% | 0.552 | 0.547 | 0.549 |
| dandelions | 66.16% | 0.668 | 0.662 | 0.665 |
| sunflowers | 51.71% | 0.523 | 0.517 | 0.520 |
| **Average** | **58.54%** | **0.581** | **0.575** | **0.578** |

**Observations:**
- Dandelions are best recognized due to distinctive texture (pinnate leaves, numerous narrow ligulate flowers)
- Main confusion pair: roses ↔ sunflowers (yellow color and radial symmetry cause misclassification)
- Removing dandelions from three-class problem increases binary accuracy to ~66%

## Installation

### Requirements

```bash
pip install numpy torch pillow opencv-python scikit-image matplotlib scikit-learn
```

### Dataset Structure

For training, organize images in the following structure:
flowers/
├── roses/
│   ├── rose1.jpg
│   ├── rose2.jpg
│   └── ...
├── dandelions/
│   ├── dandelion1.jpg
│   ├── dandelion2.jpg
│   └── ...
└── sunflowers/
    ├── sunflower1.jpg
    ├── sunflower2.jpg
    └── ...

## Key Advantages

- **No hyperparameter tuning** required (unlike SVM with RBF kernels)
- **Memory efficient** - updates only two components of λ per iteration
- **Theoretically grounded** - proven linear convergence rate
- **Interpretable** - separation vector has geometric meaning
- **Robust** to class imbalance with median-based threshold calibration

## Performance Characteristics

### Time Complexity
- Per iteration: `O(n * max(d_k))` operations
- Total iterations: typically `500-1000` for cyclic scheme
- Feature extraction: ~0.5-1.0 seconds per image (128×128 resolution)

### Space Complexity
- `O(n * Σd_k)` for storing all points
- `O(n)` for the separation vector `v`

## Limitations and Future Work

1. **Roses vs Sunflowers confusion**: Current features insufficient to reliably distinguish these classes. Solutions:
   - Add radial symmetry analysis
   - Increase image resolution for fine detail preservation
   - Use specialized descriptors for ligulate flowers vs petals

2. **Dimensionality**: Raw feature vector size is 35494 dimensions. PCA reduces to 128 components, but some information is lost.

3. **Hierarchical classification**: Consider using dandelions as an "anchor" class for two-stage classification.

## References

1. Tamasyan G.Sh. Generalization of the MDM method: presentation. 2025.
2. Malozemov V.N. MDM method turns 50 // CNSA & NDO Seminar. November 10, 2021.
3. Solovyova N.A. Linear convergence rate of the MDM method // CNSA & NDO Seminar. September 29, 2016.

## License

This project is submitted as coursework for the "Numerical Methods" discipline at Saint Petersburg State University of Economics (SPbGEU).

## Author

**Daria Vazhova**  
Applied Mathematics and Computer Science (01.03.02)  
UNECON, Faculty of Economics, Finance and Information Technologies  
Supervisor: N.A. Solovyova, Ph.D., Associate Professor

---

*Saint Petersburg, 2026*
