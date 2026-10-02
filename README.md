# PBL-turbofan-predictive-maintenance
# Predictive Maintenance of Turbofan Engines Using Deep Learning

An aerospace predictive maintenance (PdM) pipeline built to forecast impending mechanical failures in turbofan jet engines using multivariate thermodynamic sensor telemetry (NASA C-MAPSS FD001).

## Project Objectives (IEEE Rubric Compliance)
This repository fulfills the strict requirements for sequential data forecasting and anomaly detection:
* **CO3 (Baseline vs Advanced):** Implements a Vanilla RNN baseline to empirically demonstrate the vanishing gradient problem over 50-cycle sequence windows.
* **CO1 (Gate-Level Mechanics):** Transitions to a Stacked LSTM architecture with explicit gating mechanisms (Forget, Input, Output) to capture long-range degradation dependencies, achieving 98.4% anomaly recall.
* **CO4 (Production Deployment):** Conducts a runtime/accuracy trade-off analysis to select a lighter-weight Robust GRU model. The GRU merges the hidden and cell states into Update/Reset gates, providing a streamlined memory footprint. 

## Repository Structure
* `Model_Training_Pipeline.ipynb`: The core Jupyter Notebook containing the data engineering pipeline (Min-Max scaling, chronological sliding windows), hyperparameter experimentation, model training (RNN, LSTM, GRU), and visual learning curve diagnostics.
* `app.py`: The production script that launches the interactive SCADA web dashboard (built with Gradio). It features real-time telemetry streaming, live Health Index calculations, and Monte Carlo dropout passes for Epistemic Uncertainty Validation.
* `train_FD001.txt`: The raw NASA C-MAPSS telemetry dataset.

## Technical Stack
* **Deep Learning Framework:** TensorFlow / Keras
* **Data Processing:** Pandas, NumPy, Scikit-learn
* **Visualization:** Matplotlib
* **Deployment:** Gradio
