# ==============================================================================
# Makefile for the Traffic Forecasting Project
# ==============================================================================

# --- Configuration ---
PYTHON := $(shell command -v python3 2>/dev/null || echo python)

# Định nghĩa các file dữ liệu
COMBINED_DATA_TRAIN_VAL    := data/processed/train_val_combined.parquet
AUGMENTED_DATA_TRAIN_VAL   := data/processed/train_val_augmented.parquet
EXTREME_DATA_TRAIN_VAL     := data/processed/train_val_extreme_augmented.parquet

# File cuối cùng dùng để train và test (để lấy scaler)
FINAL_DATASET := $(EXTREME_DATA_TRAIN_VAL)

# --- Main Commands ---

.PHONY: all
all: process train test ## Run the full pipeline: process, train, and test

.PHONY: setup
setup: ## Install all required Python packages
	@echo ">> Installing dependencies from requirements.txt..."
	$(PYTHON) -m pip install -r requirements.txt
	@echo ">> Setup complete."

.PHONY: process
process: combine augment extreme-augment ## Run the full data processing and augmentation pipeline
	@echo ">> Full data processing pipeline complete."
	@echo ">> Final training data is ready at: $(FINAL_DATASET)"

.PHONY: combine
combine: ## Step 1: Combine raw train/val runs into a single Parquet file
	@echo ">> Combining raw train/val runs..."
	$(PYTHON) scripts/combine_runs.py --runs-dir data/runs --output-file $(COMBINED_DATA_TRAIN_VAL) --validate

.PHONY: augment
augment: $(COMBINED_DATA_TRAIN_VAL) ## Step 2: Apply advanced augmentation
	@echo "\n>> Applying advanced augmentation..."
	$(PYTHON) scripts/augment_data_advanced.py --input-dataset $(COMBINED_DATA_TRAIN_VAL) --output-dataset $(AUGMENTED_DATA_TRAIN_VAL)

.PHONY: extreme-augment
extreme-augment: $(AUGMENTED_DATA_TRAIN_VAL) ## Step 3: Apply extreme augmentation for a dense time series
	@echo "\n>> Applying EXTREME augmentation..."
	$(PYTHON) scripts/augment_extreme.py --input $(AUGMENTED_DATA_TRAIN_VAL) --output $(EXTREME_DATA_TRAIN_VAL)

.PHONY: train
train: $(FINAL_DATASET) ## Train the Graph WaveNet model using the final processed data
	@echo ">> Starting model training with data from $(FINAL_DATASET)..."
	$(PYTHON) train.py # train.py should use FINAL_DATASET
	@echo ">> Training complete. Best model saved to best_graphwavenet_model.pth"

.PHONY: test
test: best_graphwavenet_model.pth ## Evaluate the trained model on the holdout test set
	@echo ">> Evaluating model on the holdout test set..."
	$(PYTHON) test.py # test.py uses FINAL_DATASET to get scaler and 'data/runs_holdout_test' for evaluation
	@echo ">> Evaluation complete."

.PHONY: clean
clean: ## Remove all generated files (processed data, models, cache, etc.)
	@echo ">> Cleaning generated files..."
	@echo "   - Removing Parquet files..."
	-rm -f data/processed/*.parquet
	-del /Q data\processed\*.parquet 2>nul
	@echo "   - Removing model artifacts and plots..."
	-rm -f *.pth *.png
	-del /Q *.pth *.png 2>nul
	@echo "   - Removing Python cache..."
	-find . -type d -name "__pycache__" -exec rm -rf {} +
	-for /d /r . %d in (__pycache__) do @if exist "%d" rd /s /q "%d"
	@echo ">> Cleanup complete."

.PHONY: check-data
check-data: ## Validate the integrity of the raw 'runs' data directory
	@echo ">> Checking integrity of raw train/val runs..."
	$(PYTHON) scripts/check_runs_integrity.py --input data/runs
	@echo "\n>> Checking integrity of raw holdout test runs..."
	$(PYTHON) scripts/check_runs_integrity.py --input data/runs_holdout_test

.PHONY: help
help: ## Display this help message
	@echo "Makefile for the Traffic Forecasting Project"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'