
# Assembly Line Monitoring Frontend - Task Detection UI

This project provides a frontend interface for monitoring tasks in an assembly line using machine learning-based task detection. It visualizes task durations, cycle times, and missing actions, helping users analyze and optimize assembly line performance.

## Project Structure
- `main_detection_ui.py`: Main PyQt6 UI for task detection and visualization.
- `lightweight_ui.py`: Lightweight PyQt6 UI version.
- `task_detection_v3_5.py`: Core logic for task detection, using YOLOv8 segmentation models.
- `plot_tasks.py`: Generates plots for task durations and cycle times.
- `requirements.txt`: Python dependencies for the project.
- `task_times.csv`: Stores task timing data.
- `missing_actions.log`: Log file for missing actions.
- `plots/`: Contains generated plot images (Gantt charts, cycle times, etc.).
- `best_AUG_04.pt`: Example trained model weights for YOLOv8 segmentation.
- `training_files/`: Contains training scripts, datasets, and model runs.
  - `load_dataset_from_roboflow.py`: Script to download dataset from Roboflow.
  - `train_obj_detection_along_with_boxes.py`: Script to train YOLOv8 segmentation models.
  - `Red-Box-Detection-7/`: Sample dataset in YOLOv8 format (images, labels, data.yaml, README).
  - `runs/segment/`: Contains multiple training runs, results, and model weights.
- `__pycache__/`: Compiled Python files.

## Features
- Detects and visualizes tasks in an assembly line using deep learning (YOLOv8 segmentation).
- Trains and evaluates custom models on annotated datasets (Roboflow integration).
- Generates Gantt charts, cycle time plots, and confusion matrices for analysis.
- Logs missing actions and stores task timing data in CSV format.
- Includes sample datasets and training scripts for reproducibility.

## Dataset
- Dataset: Red Box Detection v7 (YOLOv8 format, 480 images, 8 classes)
- Downloaded and managed via Roboflow (`load_dataset_from_roboflow.py`)
- Augmentations: brightness, blur, salt & pepper noise
- See `training_files/Red-Box-Detection-7/README.dataset.txt` and `README.roboflow.txt` for details

## Model Training
- Training scripts provided in `training_files/`
- Example weights and checkpoints in `runs/segment/*/weights/`
- Models trained using YOLOv8 segmentation (ultralytics)
- Configurable via `data.yaml` and training scripts

## Getting Started
1. **Install dependencies:**
   ```cmd
   pip install -r requirements.txt
   ```
2. **Download dataset (optional):**
   ```cmd
   python training_files/load_dataset_from_roboflow.py
   ```
3. **Train a model (optional):**
   ```cmd
   python training_files/train_obj_detection_along_with_boxes.py
   ```
4. **Run the main UI:**
   ```cmd
   python main_detection_ui.py
   ```
5. **View generated plots:**
   Check the `plots/` directory for output images.

## Usage
- Use the UI to monitor and analyze assembly line tasks in real time.
- Review logs and plots to identify bottlenecks and optimize processes.
- Train and evaluate new models using provided scripts and datasets.

## Requirements
- Python 3.12+
- See `requirements.txt` for required packages (streamlit, ultralytics, pyqt6, pyrealsense2, etc.)

## License
This project is for internal use and research purposes.
