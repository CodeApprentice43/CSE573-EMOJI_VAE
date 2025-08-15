# CSE573-EMOJI_VAE

## Disclaimer
    This document assumes that the code is being ran locally since we are submitting the zip file directly. For best results, run on a CUDA GPU as the code is setup to detect CUDA automatically during training. 
# Setup

1. **Create and activate virtual python environment**
    ```
   python3 -m venv venv
   source venv/bin/activate
   ```
2. **Install dependencies**  
   ```
   pip install -r requirements.txt
   ```

## Data Preparation
 The codebase already contains all the data in training_data/emojis. However if you wish to add more data please add them to /orginal_data and then run the command ```python image_proc.py``` to process the new dataset for training. If you are not adding new data you may skip this step.

## Usage

Run the training script:

```
python training.py
```

You can customize parameters:

```
python training.py --epochs 1000 --batch_size 32 --latent_dim 32 --data_dir ./training_data --output_dir ./outputs --device cuda
```

## Outputs
    - All outputs are saved in the /outputs directory (gets created once training starts)
    - Model checkpoints and loss history are saved in the `outputs` directory.
    - Training loss and metric plots are saved as `loss_and_metrics.png`.
    - Reconstructions and generations are saved every 10 epochs in /generations and /reconstructions respectively
    - original_images.png contains the original training batch for comparsion 

