import glob
import torchvision.transforms as trForms
from torch.utils.data import Dataset
import os
from PIL import Image
import random
root = "/data/zhuowei/datasets/cyclegan/datasets/apple2orange"

TRANSFORMS_train = [
    trForms.Resize(int(256 * 1.12), Image.BICUBIC),  #尺寸放大
    trForms.RandomCrop(256),  # 随机剪裁到256
    trForms.RandomHorizontalFlip(),
    trForms.ToTensor(),
    trForms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
]

class Apple2OrangeDataset(Dataset):
    def __init__(self, root = "", transforms = None, model = "train"):
        super(Apple2OrangeDataset, self).__init__()
        self.img_A_paths = glob.glob(os.path.join(root, model, "A/*"))
        self.img_B_paths = glob.glob(os.path.join(root, model, "B/*"))
        self.transfroms = trForms.Compose(transforms)
        


    def __getitem__(self, index):
        path_A = self.image_A_paths[index % len(self.img_A_paths)]  # 取余
        path_B = random.choice(self.img_B_paths)

        def loader(path):
            return Image.open(path)
        img_A = loader(path_A)
        img_B = loader(path_B)
        img_A = self.transfroms(img_A)
        img_B = self.transfroms(img_B)

        return {"A": img_A, "B":img_B}
    
    def __len__(self):
        return max(len(self.img_A_paths), len(self.img_B_paths))