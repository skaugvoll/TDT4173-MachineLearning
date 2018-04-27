import numpy as np
import os, sys
from PIL import Image

class DataGenerator():
    def __init__(self, dataset="../chars74k-lite", normalized=True):
        self.dataset = dataset
        self.data = []
        self.labels = []
        self.image_names = []
        self.img_dimensions = []

        directories = [x[1] for x in os.walk(self.dataset)][0]
        print("Len: {} :: subDirs: {}".format(len(directories), directories))

        for dir in directories:
            _,_ = self._generate_data(dataset +"/" + dir, normalized=normalized)

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)



    def _generate_data(self, subdir, normalized):
        old_length = len(self.data)

        for file in os.listdir(subdir):
            if file.endswith(".jpg"):
                self.image_names.append("{}/{}".format(subdir,file))
                img = Image.open("{}/{}".format(subdir, file))
                self.img_dimensions.append(img.size)
                pixels = list(img.getdata())
                if normalized:
                    pixels = [float(x)/255 for x in pixels]

                self.data.append(pixels)

        label = self.char_to_int(subdir.split("/")[-1])

        for x in range((len(self.data) - old_length)):
            self.labels.append(label)


        return np.array(self.data), np.array(self.labels)


    def shuffle_data(self):
        data = self.get_zipped_data()
        np.random.shuffle(data)
        cases = []
        labels = []
        for case, label in data:
            cases.append(case)
            labels.append(label)

        self.data = np.array(cases)
        self.labels = np.array(labels)


    def char_to_int(self, char):
        return int(ord(char)) - 97

    def int_to_char(self, int):
        return chr(int + 97)

    def get_dimensions(self, case_number):
        return self.img_dimensions[case_number]

    def get_cases(self):
        return self.data

    def get_labels(self):
        return self.labels

    def get_data(self):
        return self.data, self.labels

    def get_image_names(self):
        return self.image_names

    def get_zipped_data(self):
        return [[c,y] for c,y in zip(self.data, self.labels)]


    def get_training_partition(self, percentage=.8):
        if percentage > 1 or percentage < 0:
            print("Invalid percantage; range: [0,1]")
            sys.exit(0)

        number_of_cases = int(len(self.data) * percentage)
        return self.data[:number_of_cases], self.labels[:number_of_cases]

    def get_testing_partition(self, percentage=.2):
        if percentage > 1 or percentage < 0:
            print("Invalid percantage; range: [0,1]")
            sys.exit(0)

        percentage = 1 - percentage
        number_of_cases = int(len(self.data) * percentage)
        return self.data[number_of_cases:], self.labels[number_of_cases:]


    def make_onehot_labels(self):
        labels = []
        for l in self.labels:
            case_onehot = []
            for i in range(0, 26):
                if i == l:
                    case_onehot.append(1)
                else:
                    case_onehot.append(0)
            labels.append(case_onehot)

        self.labels = np.array(labels)

def main():
    dg = DataGenerator()

    # a = len(dg.get_cases())
    # b,_ = dg.get_training_partition()
    # b = len(b)
    # print("\n{}\n{}".format(a,b))


    images = dg.get_image_names()
    print(images)






if __name__ == "__main__":
    main()


