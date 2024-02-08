# How to run DenseFuse Network?

1. Make a Python environment with all the required libraries
2. All the required parameters to run the experiment are provided as command line arguments using ```argparse``` in ```train.py``` file. They are explained below.

* ```device```: If a NVIDIA GPU is available use ```cuda``` otherwise use ```cpu```(If processor is not strong enought to handle the load, the program may crash)
* ```ir_img```: Path to the folder where NIR images are stored
* ```rgb_img```: Path to the folder where RGB images are stored
* ```epochs```: Number of Epochs 
* ```batch_size```: Batch size (Number of images in each step)
* ```exp_name```: Name of the experiment. A folder with the same name will be created to store the results. It is recommended that, weights assign to losses should be reflected in the experiment name to maintain an unique name for each experiment 

3. Run ```python train.py``` command in command prompt with valid arguments.

<b>Note: </b>
The weights assigned to each loss is <u>hard coded</u> in the ```train.py``` file which needs to be changed manually with each experiment. However, these losses can be easily incorporated as command line arguments. 