import numpy as np
import os
from PIL import Image

class dtaset():
    def __init__(self,img_dir):
        self.img_dir = img_dir
        self.list_files = os.listdir(self.img_dir)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.img_dir,img_file)
        image = np.array(Image.open(img_path))
        w = len(image[0]) // 2
        input_image = image[:,w:,:]
        target_image = image[:,:w,:]

        return input_image,target_image

path = "../data/train"
test = dtaset(path)
idx = np.random.randint(10000, size=10)
for i in range(0,3):
    input_image,target_image = test.__getitem__(idx[i])
    data_a = Image.fromarray(input_image)
    data_b = Image.fromarray(target_image)
    
    data_a.show()
    data_b.show()