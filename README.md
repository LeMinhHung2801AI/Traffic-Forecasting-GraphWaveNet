# Graph WaveNet for Traffic Speed Forecasting in Ho Chi Minh City

This project implements a state-of-the-art Graph WaveNet model for real-time traffic speed forecasting. It features a robust, leak-free data processing pipeline that includes advanced and extreme data augmentation techniques to create a high-quality training dataset from sparse initial data.

The system is designed to predict traffic speeds for multiple future time steps (up to 3 hours ahead) and has demonstrated exceptional performance on a held-out test set of real-world traffic data from Ho Chi Minh City.

## Key Features

- **Advanced Data Processing:** A multi-step pipeline to combine, clean, and validate raw traffic data from multiple collection runs.
- **Leak-Free Data Augmentation:** Implements safe data augmentation strategies (pattern variations, noise injection, synthetic weather, etc.) exclusively on the training set to prevent data leakage.
- **Spatio-Temporal Modeling:** Utilizes the Graph WaveNet architecture to capture both spatial dependencies (via graph convolutions and a self-adaptive adjacency matrix) and temporal patterns (via dilated causal convolutions).
- **Holdout Validation:** Employs a strict holdout test set methodology to ensure a scientifically valid and trustworthy evaluation of the model's real-world performance.
- **Automated Workflow:** Includes a `Makefile` for easy, one-command execution of the entire data processing, training, and evaluation pipeline.

## Project Structure

```
.
├── config/                 # Configuration files (e.g., augmentation)
├── data/
│   ├── runs/               # Raw data for training and validation
│   └── runs_holdout_test/  # Raw data reserved exclusively for final testing
│   └── processed/          # Generated Parquet files
├── models/
│   └── graphwavenet.py     # Graph WaveNet model architecture
├── scripts/
│   ├── combine_runs.py     # Combines raw runs into a single dataset
│   ├── augment_*.py        # Data augmentation scripts
│   └── check_runs.py       # Data integrity checking tool
├── utils/
│   ├── canonical_data.py   # Defines the canonical data structure
│   ├── adapters.py         # Adapts canonical data for the model
│   └── validation/         # Data validation utilities
├── train.py                # Script to train the model
├── test.py                 # Script to evaluate the final model
├── Makefile                # Automation for project workflows
├── requirements.txt        # Python dependencies
└── README.md
```

## Final Performance Results

The model was trained on an extensively augmented dataset and evaluated on a completely unseen, non-augmented holdout test set.

**Overall Test Set Metrics:**

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **MAE** | **0.91 km/h** | The model's predictions are, on average, off by less than 1 km/h. |
| **RMSE** | **1.53 km/h** | Low root mean squared error indicates few very large prediction errors. |
| **R² Score** | **0.9266** | The model explains **92.66%** of the variance in traffic speed. |
| **MAPE** | **5.67%** | The average percentage error is very low. |

**Performance by Prediction Horizon:**

The model demonstrates stable and highly accurate performance even for long-range forecasts. For a 15-minute forecast, the MAE is only **0.93 km/h**.

## Getting Started

### Prerequisites
- Python 3.8+
- An NVIDIA GPU with CUDA is highly recommended for training.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd Graph-WaveNet
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    The `Makefile` can handle this for you.
    ```bash
    make setup
    ```
    *Note: The `torch` installation might need to be adjusted based on your specific CUDA version. See instructions at [pytorch.org](https://pytorch.org/).*

### Running the Full Pipeline

The entire workflow, from data processing to evaluation, can be run with a single command.

1.  **Place Raw Data:**
    - Place your training/validation run folders inside `data/runs/`.
    - Place your holdout test run folders inside `data/runs_holdout_test/`.

2.  **Run the pipeline:**
    ```bash
    make all
    ```
    This command will automatically:
    - **Process** the data (`make process`).
    - **Train** the model (`make train`).
    - **Test** the final model (`make test`).

### Individual Commands

You can also run each step individually using the `Makefile`.

-   **Check data integrity:**
    ```bash
    make check-data
    ```
-   **Run the full data processing pipeline:**
    ```bash
    make process
    ```
-   **Train the model (after processing):**
    ```bash
    make train
    ```
-   **Evaluate the model (after training):**
    ```bash
    make test
    ```
-   **Clean up all generated files:**
    ```bash
    make clean
    ```

For a full list of commands, run `make help`.