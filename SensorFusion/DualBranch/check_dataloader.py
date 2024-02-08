import glob
from PIL import Image
import os

root1 = "/home/deshpand/thesis_rr/rgb_nir_fusion/ImageFusion_Dualbranch_Fusion/data/ir"
root2 = "/home/deshpand/thesis_rr/rgb_nir_fusion/ImageFusion_Dualbranch_Fusion/data/rgb"
files = glob.glob(os.path.join(root1, "*.png")) + glob.glob(
    os.path.join(root2, "*.png")
)

print(files[-1])
