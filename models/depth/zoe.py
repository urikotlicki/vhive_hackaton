import torch
from PIL import Image
import numpy as np

from models.model_runner import ModelRunner

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ZoeDepth(ModelRunner):
    def __init__(self, model_name: str = "ZoeD_N"):
        self.repo = "isl-org/ZoeDepth"

        # Zoe_N
        self.model = torch.hub.load(self.repo, "ZoeD_N", pretrained=True)

        # # Zoe_K
        # self.model = torch.hub.load(repo, "ZoeD_K", pretrained=True)

        # # Zoe_NK
        # self.model = torch.hub.load(repo, "ZoeD_NK", pretrained=True)

    #TODO: Define Image class
    def pre_process(self, image_data: Image):
        self.image = image_data.rgb

    def run(self):
        zoe = self.model.to(DEVICE)
        self.depth_map = zoe.infer_pil(self.image)  # as numpy

    def post_process(self):
        return self.depth_map


# Local file

# image = Image.open("./dataset/1.jpg").convert("RGB")  # load
# depth_numpy = zoe.infer_pil(image)  # as numpy
# np.save("depth_1.npy", depth_numpy)

# image = Image.open("./dataset/2.jpg").convert("RGB")  # load
# depth_numpy = zoe.infer_pil(image)  # as numpy
# np.save("depth_2.npy", depth_numpy)

# depth_pil = zoe.infer_pil(image, output_type="pil")  # as 16-bit PIL Image

# depth_tensor = zoe.infer_pil(image, output_type="tensor")  # as torch tensor

# import matplotlib.pyplot as plt
# fig = plt.figure()
# plt.imshow(image)
# fig.show()

# # Tensor
# from zoedepth.utils.misc import pil_to_batched_tensor
# X = pil_to_batched_tensor(image).to(DEVICE)
# depth_tensor = zoe.infer(X)



# # From URL
# from zoedepth.utils.misc import get_image_from_url

# # Example URL
# URL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS4W8H_Nxk_rs3Vje_zj6mglPOH7bnPhQitBH8WkqjlqQVotdtDEG37BsnGofME3_u6lDk&usqp=CAU"


# image = get_image_from_url(URL)  # fetch
# depth = zoe.infer_pil(image)

# # Save raw
# from zoedepth.utils.misc import save_raw_16bit
# fpath = "/path/to/output.png"
# save_raw_16bit(depth, fpath)

# # Colorize output
# from zoedepth.utils.misc import colorize

# colored = colorize(depth)

# # save colored output
# fpath_colored = "/path/to/output_colored.png"
# Image.fromarray(colored).save(fpath_colored)