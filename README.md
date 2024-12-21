Based on https://github.com/davidtvs/pytorch-lr-finder

The script allows you to find a consistent learning rate to train SD-base models/LoRAs.

The script uses SD 1.5 weights, which you can download from the following link:
https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main

What you need:
1. Create a folder for the script.
2. Copy lr.py inside this folder.
3. Create two subfolders: dataset and SD.
4. Place your set of pictures inside the dataset folder.
5. Inside the SD folder, create two subfolders: text_encoder and unet.
6. Download config.json and model.fp16.safetensors from https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/text_encoder and place them in the text_encoder folder.
7. Download config.json and diffusion_pytorch_model.fp16.safetensors from https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/unet and place them in the unet folder.
8. Open a command prompt in the folder containing lr.py and run the script.

Result:
As a result, you will get a figure showing the relationship between loss and learning rate. To find the optimal learning rate for your dataset, identify the first decreasing loss line before the plateau. The midpoint of this line is approximately the ideal learning rate (based on the original repo and another method using EveryDream2 with a validation-loss-based LR finder, which you can read about at the following link:
https://medium.com/@damian0815/the-lr-finder-method-for-stable-diffusion-46053ff91f78 ).

Note: The "Suggested learning rate" displayed in the command prompt may not work as intended.

You can manually adjust the script's settings in lines 233–250 and 290.

Args: lr.py [-h] [--model_path MODEL_PATH] [--dataset_path DATASET_PATH] [--start_lr START_LR] [--end_lr END_LR]
             [--batch_size BATCH_SIZE] [--num_iter NUM_ITER] [--seed SEED]

options:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        Path to SD model directory
  --dataset_path DATASET_PATH
                        Path to dataset directory
  --start_lr START_LR   Starting learning rate
  --end_lr END_LR       Final learning rate
  --batch_size BATCH_SIZE
                        Batch size
  --num_iter NUM_ITER   Number of iterations
  --seed SEED           Random seed for reproducibility
