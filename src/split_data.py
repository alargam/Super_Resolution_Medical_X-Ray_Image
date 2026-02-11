import random
import glob
import pickle

def split_data():
    all_images_list = glob.glob(f"./data/*/*/*.png", recursive=False)
    random.shuffle(all_images_list)
    train_images = all_images_list[:90000]
    test_images = all_images_list[90000:]
    with open("./data/train_images.pkl", "wb") as fp:
        pickle.dump(train_images, fp)
    with open("./data/val_images.pkl", "wb") as fp:
        pickle.dump(test_images, fp)

if __name__ == "__main__":
    split_data()
