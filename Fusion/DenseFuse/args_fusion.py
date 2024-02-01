
class args():

	# training args
	epochs = 50 #"number of training epochs, default is 2"
	batch_size = 2 #"batch size for training, default is 4"

	# According to the dataset choosen, un-comment the following lines.
	# Uni-dataset

	# NIR and RGB image dataset path
	# dataset_rgb = "/home/deshpand/noadsm/datasets/Uni-dataset/final_dataset/train/data/rgb"#"/home/deshpand/thesis_rr/datasets/our_forest/0001/train/rgb""#"MSCOCO 2014 path"
	# dataset_ir = "/home/deshpand/noadsm/datasets/Uni-dataset/final_dataset/train/data/nir"#"/home/deshpand/noadsm/datasets/Uni-dataset/002/nir"

	# NDVI and RGB image dataset path
	# dataset_rgb = "/home/deshpand/thesis_rr/datasets/our_forest/0001/train/rgb"
	# dataset_ir = "/home/deshpand/thesis_rr/datasets/our_forest/0001/train/ndvi"

	test_folder_path = "/home/deshpand/noadsm/datasets/Uni-dataset/final_dataset/test/data"#"/home/deshpand/thesis_rr/datasets/our_forest/0001/test"#"/home/deshpand/thesis_rr/datasets/our_forest/0001/test"  # This is the folder where all the images required to test the model are stored.
	# Freiburg Forest dataset
	# dataset_rgb = "/home/deshpand/noadsm/datasets/freiburg_forest/resized/train/rgb"
	# dataset_ir = "/home/deshpand/noadsm/datasets/freiburg_forest/resized/train/nir"
	output_path = "./outputs/Uni-dataset_ssim_wt_10000_grad_wt_1e-4_test"
	HEIGHT = 512
	WIDTH = 512
	channels = 3  # 1 means gray scale and 3 means rgb image.

	save_model_dir = "models" #"path to folder where trained model will be saved."
	save_loss_dir = "models/loss"  # "path to folder where trained model will be saved."

	image_size = 256 #"size of training images, default is 256 X 256"
	cuda = 1 #"set it to 1 for running on GPU, 0 for CPU"
	seed = 42 #"random seed for training"
	ssim_weight = [1,10,100,1000,10000, 1000000]
	gradient_weight = 0.0001
	ssim_path = ['1e0', '1e1', '1e2', '1e3', '1e4', '1e6']

	lr = 1e-3 #"learning rate, default is 0.001"
	#lr_light = 1e-2  # "learning rate, default is 0.001"
	log_interval = 5 #"number of images after which the training loss is logged, default is 500"
	resume = None
	resume_auto_en = None
	resume_auto_de = None
	resume_auto_fn = None

	# for test Final_cat_epoch_9_Wed_Jan__9_04_16_28_2019_1.0_1.0.model
	model_path_gray = "models/best_models/Uni-dataset_ssim_wt_10000_grad_wt_0.0001_23.model"#"models/best_models/grad_loss_0.0001_1_channel_lr_1e_n2_47.model"#"./models/best_models/ssim_weigth_10000_lr_1e_n2_49.model"#"./models/1e2/Uni_dataset_Final_epoch_50_Fri_Jun__2_09_00_28_2023_1e2.model"  #"./models/densefuse_gray.model"
	model_path_rgb = "models/best_models/ssim_weigth_10000_lr_1e_n2_49.model"#"./models/densefuse_rgb.model"

	# The following will be the name of the experiment. Change this everytime to make the saved data distinguishable.
	exp_name = 'Uni-dataset_NDVI_ssim_wt_10000_lr_1e-3_50'
	dataset = ['Freiburg_Forest', 'Uni-dataset']
