import numpy as np
import cv2
import os

def load_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(224,224))
    img = np.reshape(img,(1,224,224))
    img = img.astype("float32")/255
    img = cv2.GaussianBlur(img,(9,9), 0)
    return img

classify_list = ['8PSK','BPSK' ,'PAM4' ,'QAM16','QAM64','QPSK']

def load_dataset():
    # train:val 8:2 也就是40:10
    dataset_path = './works/dataset1/dataset/'
    classes = os.listdir(dataset_path)
    classify = {
        '8PSK' :[1,0,0,0,0,0],
        'BPSK' :[0,1,0,0,0,0],
        'PAM4' :[0,0,1,0,0,0],
        'QAM16':[0,0,0,1,0,0],
        'QAM64':[0,0,0,0,1,0],
        'QPSK' :[0,0,0,0,0,1],
        }
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for clazz in classes:
        img_dir = dataset_path+clazz+'/'
        imgs = os.listdir(img_dir)
        group = dict()
        print(f'loading gorup:{clazz}...')
        for name in imgs:
            dB = int(name.split('dB')[0])
            if group.__contains__(dB):
                group[dB].append(name)
            else:
                group[dB] = [name]
        for dB,img_ns in group.items():
            if dB<=3: 
                continue
            img_train = img_ns[:45]
            img_test = img_ns[-5:]
            for img_p in img_train:
                img = load_img(img_dir + img_p)
                x_train.append(img)
                y_train.append(classify[clazz])
                # 数据增强
                if dB >= 10 :
                    tmp = np.flip(img,1)
                    tmp = np.flip(tmp,2)
                    x_train.append(tmp)
                    y_train.append(classify[clazz])

            for img_p in img_test:
                img = load_img(img_dir + img_p)
                x_test.append(img)
                y_test.append(classify[clazz])
    x_train = np.stack(x_train)
    y_train = np.reshape(y_train,(len(y_train),6,1))
    x_test = np.stack(x_test)
    y_test = np.reshape(y_test,(len(y_test),6,1))
    return (x_train,y_train),(x_test,y_test)