import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
train_dir = "../loss/resnet/train_resnet_loss.txt"
valid_dir = "../loss/resnet/valid_resnet_loss.txt"

def open_file(dir):
    with open(dir, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")
    return data
if __name__ == "__main__":
    data = open_file(train_dir)
    data = np.array(data)
    data = data.astype(np.float)
    x = np.arange(1,len(data)+1)
    print(data)
    print(x)
    plt.xlabel('Epoch')
    plt.ylabel('Train_Loss')
    plt.plot(x, data, linewidth=1, linestyle="solid", label="train loss")
    plt.legend()
    plt.title('Train Loss Resnet')
    #plt.savefig('../loss/lstm/train_resnet_loss.jpg')
    plt.savefig('../loss/resnet/train_resnet_loss.jpg')
    plt.show()
    
    