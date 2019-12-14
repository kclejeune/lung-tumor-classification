from torch.utils.data.dataset import Dataset
from glob import glob
from PIL import Image
from os.path import join
from math import log10

SLICES_PER_STUDY = 65


class LIDCDataset(Dataset):
    def __init__(self, data_path="./data/lidc"):
        self.data_path = data_path
        self.count = SLICES_PER_STUDY * len(glob(data_path + "/*"))
        self.labels = self.get_labels()

    def __getitem__(self, index):
        study_id = int(index / SLICES_PER_STUDY) + 1
        formated_study_num = self.configure_number(study_id)

        image_num = index % SLICES_PER_STUDY

        study_path = join(self.data_path, "LIDC-IDRI-{}".format(formated_study_num))
        img = Image.open(join(study_path, "{}.png".format(image_num)))

        return (img, self.labels[index])

    def configure_number(self, num, length=4):
        dig = int(log10(num) + 1)
        zero_dig = length - dig

        zeros = "".join([str(0) for x in range(zero_dig)])
        return zeros + str(num)

    def get_labels(self):
        labels = []
        # Iterate over studies
        fix_study = []
        for study in sorted(glob(self.data_path + "/*")):
            with open(join(study, "annotations.txt"), "r") as f:
                # For every line in the studies annotation, read annotation
                for line in f:
                    if line is not "":
                        labels.append(int(line.split(": ")[1][:-1]))

        return labels

    def __len__(self):
        return self.count

