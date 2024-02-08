# How to run DenseFuse Network?

1. Make a Python environment with all the required libraries
2. All the required parameters to run the experiment are provided as command line arguments using ```argparse``` in ```train_vae.py``` file. They are explained below.

* ```model_name```: Name of the experiment. This will be used in WANDB and to save to create a folder in which all the results will be saved.
* ```dataset_root```: Path to the folder where RGB images are stored
* ```nir_dataset```: Path to the folder where NIR images are stored
* ```mode```: Based on this, convert the RGB image Gray scale or convert gray scale to RGB. If value is RGB, do not convert images to gray scale
* ```save_dir```: To create a folder in which the model and the results will be saved
* ```nepoch```: Number of epochs 
* ```image_size```: Resize the input images to square image of this size
* ```ch_multi```: Channel width multiplier for variational autoencoder logic
* ```device_index```: GPU index to choose one GPU from multiple
* ```latent_channels```: Number of channels of the latent space
* ```save_interval```: Number of iteration per save
* ```lr```: Learning Rate
* ```feature_scale```: Feature loss scale
* ```kl_scale```: KL penalty scale
* ```load_checkpoint```: Load from checkpoint

3. Run ```python train_vae.py``` command in command prompt with valid arguments.
