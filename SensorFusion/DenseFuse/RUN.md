# How to run DenseFuse Network?

1. Make a Python environment with all the required libraries
2. Read the fields in ```args_fusion.py``` file and make required changes such as number of epochs, batch size, RGB and NIR image paths, etc. (Please note that, all the paths provided in the file are absolute)
3. Activate the correct python environment 
4. Run ```python train_densefuse.py``` command in the command prompt. 

Note: To change the weight assigned to the SSIM loss, change the value of ```i``` in the ```args.ssim_weight[i]``` argument in the ```def train()``` function in ```train_densefuse.py``` file. This has to be done before the experiment is run. 