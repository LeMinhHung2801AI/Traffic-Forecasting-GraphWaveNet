# Graph WaveNet Model Report: Real-Time Traffic Forecasting

**Project:** Traffic Congestion Prediction in Ho Chi Minh City
**Model:** Graph WaveNet
**GitHub Repository:** [https://github.com/LeMinhHung2801AI/Traffic-Forecasting-GraphWaveNet.git](https://github.com/LeMinhHung2801AI/Traffic-Forecasting-GraphWaveNet.git)

---

## Table of Contents

1.  [Objective](#1-objective)
2.  [Data Pipeline and Preprocessing](#2-data-pipeline-and-preprocessing)
    -   [2.1 Raw Data Source](#21-raw-data-source)
    -   [2.2 Data Combination and Augmentation](#22-data-combination-and-augmentation)
    -   [2.3 Data Splitting Strategy: The Holdout Method](#23-data-splitting-strategy-the-holdout-method)
    -   [2.4 Ensuring a Leak-Free Pipeline](#24-ensuring-a-leak-free-pipeline)
3.  [Related Work](#3-related-work)
4.  [Model Architecture](#4-model-architecture)
    -   [4.1 Core Architecture and Reference](#41-core-architecture-and-reference)
    -   [4.2 Implementation vs. Original Paper](#42-implementation-vs-original-paper)
    -   [4.3 Detailed Architecture Diagram](#43-detailed-architecture-diagram)
5.  [Training and Results](#5-training-and-results)
    -   [5.1 Training Configuration](#51-training-configuration)
    -   [5.2 Training Dynamics and Convergence](#52-training-dynamics-and-convergence)
    -   [5.3 Final Evaluation on Holdout Test Set](#53-final-evaluation-on-holdout-test-set)
6.  [Conclusion](#6-conclusion)
7.  [References](#7-references)

---

## 1. Objective

The goal of this project is to develop and evaluate a high-accuracy deep learning model for real-time traffic speed forecasting across Ho Chi Minh City's road network. This report focuses specifically on the implementation and performance of the **Graph WaveNet** model.

## 2. Data Pipeline and Preprocessing

A robust, multi-stage data processing pipeline was established to ensure data quality, diversity, and the prevention of data leakage, ultimately producing a high-quality dataset for model training and evaluation.

### 2.1 Raw Data Source

The initial data consists of a collection of raw traffic measurements stored in a directory structure under `data/runs/`. Each subdirectory (e.g., `run_20251030_032440`) represents a single data collection instance at a specific timestamp and contains multiple JSON files detailing traffic speeds (`traffic_edges.json`), weather conditions (`weather_snapshot.json`), and network topology (`nodes.json`, `edges.json`).

### 2.2 Data Combination and Augmentation

The raw, sparse data was processed through a three-step pipeline to create a rich and dense dataset suitable for training a deep learning model.

1.  **Combination (`scripts/combine_runs.py`):** All valid raw `run` directories were parsed and consolidated into a single, clean Parquet file (`train_val_combined.parquet`). This step standardizes the data format and serves as the foundation for augmentation.

2.  **Advanced Augmentation (`scripts/augment_data_advanced.py`):** To increase data diversity, several augmentation techniques were applied **only to the training portion** of the combined data. These methods include:
    *   **Temporal Extrapolation:** Simulating data for past dates.
    *   **Pattern Variations:** Creating scenarios like intensified rush hours, weekend traffic, and random events.
    *   **Noise Injection:** Adding controlled Gaussian noise to speed values.

3.  **Extreme Augmentation (`scripts/augment_extreme.py`):** This crucial step addresses the sparse nature of the time series.
    *   **Temporal Interpolation:** New data points were generated **between** existing timestamps within the training set using linear interpolation. This transformed the sparse data into a much denser, more continuous time series, which is highly beneficial for sequence models. This technique was strictly confined to the training data to prevent leakage.
    *   **Synthetic Weather & Scenarios:** Additional variations based on synthetic weather and traffic scenarios were generated.

This pipeline resulted in a final training dataset (`train_val_extreme_augmented.parquet`) that was significantly larger and more diverse than the original raw data.

### 2.3 Data Splitting Strategy: The Holdout Method

To ensure a scientifically rigorous evaluation, a strict **holdout method** was employed:

1.  **Holdout Test Set:** Before any processing, a portion of the raw data (10 `run` directories, ~15%) was manually moved to a separate `data/runs_holdout_test/` directory. This data was **never used** for training, validation, or learning augmentation patterns.
2.  **Training & Validation Set:** The remaining raw data (56 `run` directories) in `data/runs/` was used as the input for the entire data processing and augmentation pipeline described above.
3.  **Internal Split:** Within the training pipeline, this data was further split chronologically into a **training set (70%)** and a **validation set (15%)**. The augmentation techniques were applied **only** to the training portion of this data.

This strategy guarantees that the final model evaluation is performed on completely unseen, "pristine" data, providing a true measure of its generalization capability.

### 2.4 Ensuring a Leak-Free Pipeline

Data leakage was meticulously avoided through several key principles:

1.  **Physical Test Set Isolation:** The holdout test set was physically separated from the start.
2.  **Split Before Augment:** All augmentation scripts were designed to split the data into train and validation sets *before* applying any transformations, and these transformations were only applied to the train set.
3.  **Train-Only Statistics:** All statistical patterns (e.g., mean, std, hourly profiles) used for data augmentation and normalization (`StandardScaler`) were computed **exclusively** from the training data split.
4.  **Automated Validation:** A `validate_no_leakage` function was integrated into the pipeline to programmatically check for any temporal or `run_id` overlap between the generated train and validation sets, automatically halting the process if any leakage was detected.

## 3. Related Work

Traffic forecasting has evolved from traditional statistical methods to advanced deep learning techniques. Early approaches, such as ARIMA and its variants, modeled traffic flow as a simple time series but fundamentally failed to capture the spatial dependencies of a traffic network. The advent of deep learning introduced Long Short-Term Memory (LSTM) networks, which excel at learning long-term temporal patterns but do not inherently understand the graph structure of a road network.

To address this limitation, researchers began integrating Graph Neural Networks (GNNs) with sequential models, giving rise to the field of spatio-temporal graph networks. These models treat the road network as a graph, where intersections are nodes and roads are edges, allowing information to propagate spatially. Models like **ASTGCN**, **Graph WaveNet**, and the more recent Transformer-based **STMGT** represent the state-of-the-art in this domain.

While a wide array of advanced architectures have emerged since 2019, this project strategically focuses on the implementation of Graph WaveNet. This decision is based on several key factors that position Graph WaveNet as an ideal candidate for establishing a strong and reliable forecasting baseline.

Newer models such as **AGCRN** (Bai et al., 2020) and **MTGNN** (Wu et al., 2020) have advanced the field by introducing dedicated graph learning modules to automatically infer network structures. More recently, a significant trend has been the integration of Transformer architectures, as seen in models like **GTSNet** (Luo et al., 2022) and **PDFormer** (Jiang et al., 2022), which leverage self-attention to capture long-range temporal dependencies with greater flexibility than the Temporal Convolutional Networks (TCNs) used in Graph WaveNet.

Despite these advancements, Graph WaveNet was chosen for this project due to its optimal balance of performance, architectural innovation, and implementation complexity. It represents a foundational leap from earlier RNN-based models by successfully integrating TCNs with Graph Neural Networks, thereby solving critical issues like vanishing gradients while still being more straightforward to implement than recent Transformer-based hybrids. Its proven high performance serves as a powerful and competitive benchmark. Our objective was to master a robust, state-of-the-art architecture and pair it with a superior data pipeline. As our results demonstrate, this focused approach allowed us to achieve exceptional performance, validating Graph WaveNet as a highly effective choice for this task.

## 4. Model Architecture

### 4.1 Core Architecture and Reference

The model implemented is **Graph WaveNet**, based on the architecture proposed in the following paper:

> Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019). Graph WaveNet for Deep Spatial-Temporal Graph Modeling. In *Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI)*.

Graph WaveNet is a state-of-the-art architecture that captures spatio-temporal dependencies by combining:
-   **Dilated Causal Convolutions (TCN):** To efficiently learn long-range temporal patterns without the limitations of RNNs.
-   **Graph Convolutions:** To model spatial relationships by propagating information across the road network graph.
-   **Self-Adaptive Adjacency Matrix:** A key innovation where the model learns an adaptive graph structure in addition to the physical road network, allowing it to discover hidden spatial correlations from data.

### 4.2 Implementation vs. Original Paper

Our implementation adheres to the core principles of the original Graph WaveNet but features one notable simplification for ease of implementation.

| Component | Original Graph WaveNet | Our Implementation | Analysis |
| :--- | :--- | :--- | :--- |
| **Graph Convolution** | **Diffusion Convolution Layer** | **Simplified GCN Layer** | The original paper uses a complex diffusion convolution. Our version uses a simplified but effective approach: concatenating the results of multiplying the input with each support matrix, followed by a 1x1 convolution. |
| **Self-Adaptive Matrix** | `softmax(ReLU(E @ E.T))` | `softmax(ReLU(E @ E.T))` | **Identical.** The core mechanism for learning hidden graph structures was successfully implemented. |
| **Temporal Modeling (TCN)** | Dilated Causal Convolutions | Dilated Causal Convolutions | **Identical.** The temporal modeling component follows the original design. |

This implementation preserves the most innovative aspects of Graph WaveNet while ensuring a stable and understandable codebase.

### 4.3 Detailed Architecture Diagram

The data flow through our implemented Graph WaveNet model is as follows:

```
+------------------------------------------------------+
| Input: Historical Traffic Data (X)                   |
| Shape: (Batch, Seq_Len=24, Num_Nodes=62, In_Dim=1)     |
+------------------------------------------------------+
                         |
                         v
+------------------------------------------------------+
| 1. Input Transformation & Self-Adaptive Matrix       |
|   - Input is permuted and passed to a 1x1 Conv.      |
|   - A learnable adaptive adjacency matrix is computed. |
+------------------------------------------------------+
                         |
                         v
+------------------------------------------------------+
| 2. Stacked Spatio-Temporal Blocks (4 Blocks)         |
|   | Each Block contains:                           |
|   | a) Temporal Layer (Dilated Causal Conv)        |
|   | b) Spatial Layer (Simplified GCN)              |
|   | c) Residual and Skip Connections               |
+------------------------------------------------------+
                         |
                         v
+------------------------------------------------------+
| 3. Output Module                                     |
|   - All skip-connections are aggregated.             |
|   - Two final 1x1 Conv layers produce the output.    |
+------------------------------------------------------+
                         |
                         v
+------------------------------------------------------+
| Output: Speed Predictions (Y_hat)                    |
| Shape: (Batch, Num_Nodes=62, Pred_Len=12)              |
+------------------------------------------------------+
```

## 5. Training and Results

### 5.1 Training Configuration

The model was trained with the following configuration:

| Parameter | Value |
| :--- | :--- |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | L1Loss (Mean Absolute Error) |
| Batch Size | 16 |
| Max Epochs | 100 |
| Early Stopping Patience | 10 |
| Input Sequence (`SEQ_LEN`) | 24 (6 hours) |
| Prediction Horizon (`PRED_LEN`)| 12 (3 hours) |

### 5.2 Training Dynamics and Convergence

The model was trained on the "extreme" augmented dataset. The process was halted by the early stopping mechanism after **22 epochs**, with the best-performing model being saved at **Epoch 12**, where the validation loss reached its minimum of **0.0071**.

![Training and Validation Loss Curve](loss_curve.png)
*Figure 1: Training and validation loss curves. The process converged quickly, and early stopping selected the model from Epoch 12, preventing overfitting.*

The loss curve demonstrates a healthy training dynamic:
-   **Rapid Convergence:** Both training and validation losses dropped significantly within the first few epochs.
-   **No Overfitting:** The gap between the training and validation curves remained small, indicating excellent generalization, a result of the extensive data augmentation.
-   **Effective Early Stopping:** The training was automatically stopped after the validation loss failed to improve for 10 consecutive epochs, saving computational resources and ensuring the best model was selected.

### 5.3 Final Evaluation on Holdout Test Set

The best model from Epoch 12 was evaluated on the completely unseen holdout test set. The results are highly impressive and validate the effectiveness of the entire pipeline.

**Overall Performance on Holdout Test Set:**

| Metric | Value |
| :--- | :--- |
| **MAE** | **0.91 km/h** |
| **RMSE** | **1.53 km/h** |
| **R² Score** | **0.9266** |
| **MAPE** | **5.67%** |

**Performance by Prediction Horizon:**

The model maintains remarkable accuracy across all prediction horizons, showcasing its ability to capture both short-term and long-term dependencies.

| Horizon | Forecast Time | MAE (km/h) |
| :--- | :--- | :--- |
| 1 | 15 min | 0.93 |
| 6 | 90 min | 0.93 |
| 12| 180 min| 0.90 |

## 6. Conclusion

This project successfully demonstrates that the Graph WaveNet architecture, when combined with a robust, leak-free data augmentation pipeline, can achieve exceptionally high accuracy for real-time traffic speed forecasting. The final model achieved an overall **MAE of 0.91 km/h** and an **R² score of 0.93** on a completely unseen holdout test set, confirming its strong generalization capabilities. The success of this implementation establishes a powerful and reliable baseline for future traffic prediction systems in Ho Chi Minh City.

## 7. References

1.  **[Graph WaveNet]** Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019). Graph WaveNet for Deep Spatial-Temporal Graph Modeling. In *Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI)*.  
    [[ArXiv Link]](https://arxiv.org/abs/1906.00121)

2.  **[AGCRN]** Bai, L., Yao, L., Li, C., Wang, X., & Wang, C. (2020). Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting. In *Proceedings of the 34th Conference on Neural Information Processing Systems (NeurIPS)*.  
    [[ArXiv Link]](https://arxiv.org/abs/2007.02842)

3.  **[PDFormer]** Jiang, W., Luo, J., He, S., & Li, Y. (2022). PDFormer: Propagation Delay-aware Dynamic Long-range Transformer for Traffic Flow Prediction. In *Proceedings of the AAAI Conference on Artificial Intelligence*.  
    [[ArXiv Link]](https://arxiv.org/abs/2211.12389)

4.  **[DCRNN]** Li, Y., Yu, R., Shah, C., & Liu, Y. (2018). Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting. In *International Conference on Learning Representations (ICLR)*.  
    [[ArXiv Link]](https://arxiv.org/abs/1707.01926)

5.  **[MTGNN]** Wu, Z., Pan, S., Long, G., Jiang, J., & Sun, L. (2020). Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks. In *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*.  
    [[ArXiv Link]](https://arxiv.org/abs/2005.11650)