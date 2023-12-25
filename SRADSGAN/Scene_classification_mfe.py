from contextlib import suppress
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import warnings
from zipfile import ZipFile
from skimage.io import imread, imsave
from tensorflow import keras
# from keras import applications
# from keras import optimizers
# from keras.models import Sequential
# from keras.layers import Activation, Dense, Dropout, Flatten
# from keras.utils import to_categorical
# from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay     #plot_confusion_matrix,
import os
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.layers.convolutional import Convolution2D
from tensorflow.python.layers.pooling import MaxPooling2D

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
'''

数据集：UC Merced Land-Use Data Set

图像像素大小为256x256，总包含21类场景图像，每一类有100张，共2100张。

下载地址：http://weegee.vision.ucmerced.edu/datasets/landuse.html
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

with suppress(FileExistsError):
    os.mkdir('data')

source_dir = "/home/zju/mfe/dataset/UCMerced_LandUse/"

# 读取分类名称
class_names = os.listdir(source_dir)

# 创建flow文件夹
flow_base = os.path.join('/home/zju/mfe/dataset', 'SceneClass')

# 创建  train/validate/test 文件夹
target_dirs = {target: os.path.join(flow_base, target) for target in ['train', 'validate', 'test']}

def create_train_validate_test_dataset():

    if not os.path.isdir(flow_base):
        # 创建新文件夹
        os.mkdir(flow_base)

        for target in ['train', 'validate', 'test']:
            target_dir = os.path.join(flow_base, target)
            os.mkdir(target_dir)
            for class_name in class_names:
                class_subdir = os.path.join(target_dir, class_name)
                os.mkdir(class_subdir)

        # 忽略 skimage.io.imsave的警告
        warnings.simplefilter('ignore', UserWarning)

        # 从 ./data/UCMerced_LandUse/Images文件夹图片复制文件到 ./data/flow/<train, validate, test>文件夹
        for root, _, filenames in os.walk(source_dir):
            if filenames:
                class_name = os.path.basename(root)

                # 随机
                filenames = np.random.permutation(filenames)
                # 随机提取65%用于训练，10%用于验证，25%用于测试
                for target, count in [('train', 65), ('validate', 10), ('test', 25)]:
                    target_dir = os.path.join(flow_base, target, class_name)
                    for filename in filenames[:count]:
                        filepath = os.path.join(root, filename)
                        image = imread(filepath)
                        basename, _ = os.path.splitext(filename)
                        # TIFF转化为png
                        target_filename = os.path.join(target_dir, basename + '.png')
                        imsave(target_filename, image)

                    filenames = filenames[count:]

        # 提示warming
        warnings.resetwarnings()

def create_downsampling_dataset(datapath, dstpath, scale):

    if not os.path.isdir(dstpath):
        # 创建新文件夹
        os.mkdir(dstpath)
        for class_name in class_names:
            class_subdir = os.path.join(dstpath, class_name)
            os.mkdir(class_subdir)

    for root, _, filenames in os.walk(datapath):
        if filenames:
            class_name = os.path.basename(root)
            # 随机
            filenames = np.random.permutation(filenames)
            target_dir = os.path.join(dstpath, class_name)
            for filename in filenames:
                filepath = os.path.join(root, filename)
                image = imread(filepath)
                image = interpolation(image, scale, interpolation='bicubic')
                basename, _ = os.path.splitext(filename)
                # TIFF转化为png
                target_filename = os.path.join(target_dir, basename + '_downsample.png')
                imsave(target_filename, image)

means = []
# for root, _, filenames in os.walk(target_dirs['train']):
for root, _, filenames in os.walk(source_dir):
    for filename in filenames:
        filepath = os.path.join(root, filename)
        image = imread(filepath)
        means.append(np.mean(image, axis=(0, 1)))
channel_means = np.mean(means, axis=0)


def get_bottleneck_features(model, dataset, preproc_func, batch_size=64):
    """
    通过VGG16预测卷积率
    从输入数据集(train/validate/test)取得 特征 X 和 标签 Y
    保存特征和标签为numpy文件

    输入:
        model: 提前训练好的深度学习模型, 不包括全连接
               比如 applications.VGG16(include_top=False, weights='imagenet')
        dataset：数据集文件名称 ['train', 'validate', 'test']
        preproc_func: 预处理函数
        batch_size: 每个batch的样本数量

    返回:
        返回 numpy数组的特征
    """

    print(f'生成"{dataset}" 特征X，标签Y')
    X_filepath = os.path.join('/home/zju/mfe/dataset/SceneClass', 'bn_' + dataset + '_X.npy')
    y_filepath = os.path.join('/home/zju/mfe/dataset/SceneClass', 'bn_' + dataset + '_y.npy')

    # 检测数据是否存在
    try:
        with open(X_filepath, 'rb') as f:
            X = np.load(f)
        with open(y_filepath, 'rb') as f:
            y = np.load(f)
            Y = keras.utils.to_categorical(y, num_classes=len(np.unique(y)))
    #取得特征和标签
    except:
        image_data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0, preprocessing_function=preproc_func)
        # image_data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0)
        image_generator = image_data_gen.flow_from_directory(target_dirs[dataset],
                                                             batch_size=batch_size,
                                                             shuffle=False,
                                                             )
        image_count = 0
        X_batches, Y_batches = [], []
        for X, Y in image_generator:
            X_batches.append(model.predict_on_batch(X))
            Y_batches.append(Y)
            image_count += X.shape[0]
            # 输入图像大于最大值中断
            if image_count >= image_generator.n:
                break

        X = np.concatenate(X_batches)
        with open(X_filepath, 'wb') as f:
            np.save(f, X)
        Y = np.concatenate(Y_batches)
        y = np.nonzero(Y)[1]
        with open(y_filepath, 'wb') as f:
            np.save(f, y)

    print(f' 图像shape {X.shape} ')
    return X, Y

def get_bottleneck_features_dataset(pre_model, dataset, dirpath, preproc_func, batch_size=64):
    """
    通过VGG16预测卷积率
    从输入数据集(train/validate/test)取得 特征 X 和 标签 Y
    保存特征和标签为numpy文件

    输入:
        model: 提前训练好的深度学习模型, 不包括全连接
               比如 applications.VGG16(include_top=False, weights='imagenet')
        dataset：数据集文件名称 ['train', 'validate', 'test']
        preproc_func: 预处理函数
        batch_size: 每个batch的样本数量

    返回:
        返回 numpy数组的特征
    """

    print(f'生成"{dataset}" 特征X，标签Y')
    X_filepath = os.path.join('/home/zju/mfe/dataset/SceneClass', 'bn_' + dataset + '_X.npy')
    y_filepath = os.path.join('/home/zju/mfe/dataset/SceneClass', 'bn_' + dataset + '_y.npy')

    # 检测数据是否存在
    try:
        with open(X_filepath, 'rb') as f:
            X = np.load(f)
        with open(y_filepath, 'rb') as f:
            y = np.load(f)
            Y = keras.utils.to_categorical(y, num_classes=len(np.unique(y)))
    #取得特征和标签
    except:
        image_data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0, preprocessing_function=preproc_func)
        # image_data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0)
        image_generator = image_data_gen.flow_from_directory(dirpath,
                                                             batch_size=batch_size,
                                                             shuffle=False,
                                                             )
        image_count = 0
        X_batches, Y_batches = [], []
        for X, Y in image_generator:
            X_batches.append(pre_model.predict_on_batch(X))
            Y_batches.append(Y)
            image_count += X.shape[0]
            # 输入图像大于最大值中断
            if image_count >= image_generator.n:
                break

        X = np.concatenate(X_batches)
        with open(X_filepath, 'wb') as f:
            np.save(f, X)
        Y = np.concatenate(Y_batches)
        y = np.nonzero(Y)[1]
        with open(y_filepath, 'wb') as f:
            np.save(f, y)

    print(f' 图像shape {X.shape} ')
    return X, Y


#  VGG19 model  include_top=false
pretrained_model = keras.applications.VGG19(include_top=False, weights='imagenet')

def build_fully_connected(input_shape, num_classes):
    """
    创建全连接模型 来训练测试 UC Merced 数据集.
    """
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=input_shape))
    model.add(keras.layers.Dense(256))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    return model

def train():
    num_classes = len(class_names)
    X, Y = dict(), dict()
    preprocess = lambda x: x - channel_means
    for dataset in ['train', 'validate', 'test']:
        # 使用VGG19从数据集里提取特征和标签0.928 VGG16
        X[dataset], Y[dataset] = get_bottleneck_features(pretrained_model, dataset, preprocess)

    # checkpoint 保存最好的模型
    filepath_checkpoint = "/home/zju/mfe/land_cover_classification/trainCNN_weights_best_multi.hdf5"
    checkpoint = ModelCheckpoint(filepath_checkpoint, monitor='val_accuracy', verbose=2,
                                 save_best_only=True, save_weights_only=False, mode='max')
    # checkpoint
    filepath_checkpoint22 = "/home/zju/mfe/land_cover_classification/trainCNN_each_epoch_multi.hdf5"
    checkpoint22 = ModelCheckpoint(filepath_checkpoint22, monitor='val_acc', verbose=2,
                                   save_best_only=False, save_weights_only=False, mode='max')
    callbacks_list = [checkpoint, checkpoint22]


    # 建立，编译，适配模型
    model = build_fully_connected(input_shape=X['train'].shape[1:], num_classes=num_classes)
    adam = keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model_fit_history = model.fit(X['train'], Y['train'], batch_size=64, epochs=100,
                                  callbacks=callbacks_list,
                                  verbose=2, validation_data=(X['validate'], Y['validate']))

    epochs = np.argmin(model_fit_history.history['val_loss']) + 1
    print(f'Stop training at {epochs} epochs')

    # 合并训练验证数据集
    X_train = np.concatenate([X['train'], X['validate']])
    Y_train = np.concatenate([Y['train'], Y['validate']])

    # 随机处理 X 和 Y
    shuffle_index = np.random.permutation(len(X_train))
    X_train = X_train[shuffle_index]
    Y_train = Y_train[shuffle_index]
    model = build_fully_connected(input_shape=X_train.shape[1:], num_classes=num_classes)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model_fit_history = model.fit(X_train, Y_train, batch_size=64, epochs=epochs,
                                  #callbacks=checkpoint22,
                                  verbose=0)

    # 预测测试数据
    y_pred = model.predict_classes(X['test'], verbose=0)

    y_test = np.nonzero(Y['test'])[1]
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model predication accuracy: {accuracy:.3f}')
    print(f'\nClassification report:\n {classification_report(y_test, y_pred)}')

    # save the result history to file
    with open("/home/zju/mfe/land_cover_classification/trainCNN__log_history.txt", 'w') as fhist:
        fhist.write(str(model_fit_history.history))

    #保存模型和权重
    model_path = "/home/zju/mfe/land_cover_classification/trainCNN_trained_model_single.h5"
    model.save(model_path)
    print('Saved single trained model at %s ' % model_path)

def evaluate(dataset, dirpath, num_classes, savepath=None):

    preprocess = lambda x: x - channel_means
    X, Y = get_bottleneck_features_dataset(pretrained_model, dataset, dirpath, preprocess, batch_size=1)
    model = build_fully_connected(input_shape=X.shape[1:], num_classes=num_classes)
    model.load_weights("/home/zju/mfe/land_cover_classification/trainCNN_trained_model_single.h5")
    # 预测测试数据
    y_pred = model.predict_classes(X, verbose=0)

    y_test = np.nonzero(Y)[1]
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)#, normalize=True
    print(f'Model predication accuracy: {accuracy:.3f}')
    print(f'\nClassification report:\n {classification_report(y_test, y_pred)}')
    print(f'\nconfusion matrix report:\n {cm}')
    display_labels = [
        "agricultural",
        "airplane",
        "baseballdiamond",
        "beach",
        "buildings",
        "chaparral",
        "denseresidential",
        "forest",
        "freeway",
        "golfcourse",
        "harbor",
        "intersection",
        "mediumresidential",
        "mobilehomepark",
        "overpass",
        "parkinglot",
        "river",
        "runway",
        "sparseresidential",
        "storagetanks",
        "tenniscourt"]

    plot_confusion_metrix(cm, display_labels=display_labels, include_values=True, cmap='viridis', xticks_rotation='horizontal',
                          values_format=None, ax=None, fontsize=8, savepath=savepath)


from skimage import transform
def interpolation(img, scale_factor, interpolation='bicubic'):
    data = list()
    H = img.shape[0]
    W = img.shape[1]
    C = img.shape[2]
    h = int(H * scale_factor)
    w = int(W * scale_factor)
    for i in range(C):
        data.append(transform.resize(img[:, :, i], [h, w], order=3, preserve_range=True)[:, :, None])
    img_out = np.concatenate(data, axis=2).astype(np.int32)
    return img_out

def plot_confusion_metrix(cm, display_labels=None, include_values=True, cmap='viridis', xticks_rotation='horizontal', values_format=None, ax=None, fontsize=1, savepath=None):
    """Plot visualization.
    Parameters
    ----------
    include_values : bool, default=True
        Includes values in confusion matrix.

    cmap : str or matplotlib Colormap, default='viridis'
        Colormap recognized by matplotlib.

    xticks_rotation : {'vertical', 'horizontal'} or float, \
                     default='horizontal'
        Rotation of xtick labels.

    values_format : str, default=None
        Format specification for values in confusion matrix. If `None`,
        the format specification is 'd' or '.2g' whichever is shorter.

    ax : matplotlib axes, default=None
        Axes object to plot on. If `None`, a new figure and axes is
        created.

    Returns
    -------
    display : :class:`~sklearn.metrics.ConfusionMatrixDisplay`
    """
    import matplotlib.pyplot as plt
    from itertools import product
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # cm = self.confusion_matrix
    n_classes = cm.shape[0]
    im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    text_ = None
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

    if include_values:
        text_ = np.empty_like(cm, dtype=object)

        # print text with appropriate color depending on background
        thresh = (cm.max() + cm.min()) / 2.0

        for i, j in product(range(n_classes), range(n_classes)):
            color = cmap_max if cm[i, j] < thresh else cmap_min

            if values_format is None:
                text_cm = format(cm[i, j], '.2g')
                if cm.dtype.kind != 'f':
                    text_d = format(cm[i, j], 'd')
                    if len(text_d) < len(text_cm):
                        text_cm = text_d
            else:
                text_cm = format(cm[i, j], values_format)

            text_[i, j] = ax.text(
                j, i, text_cm,
                ha="center", va="center",fontsize=fontsize,
                color=color)

    if display_labels is None:
        display_labels = np.arange(n_classes)
    else:
        display_labels = display_labels

    fig.colorbar(im_, ax=ax)
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=display_labels,
           yticklabels=display_labels,
           ylabel="True label",
           xlabel="Predicted label")

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)
    if savepath is not None:
        plt.savefig(savepath, dpi=500, bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':
    create_train_validate_test_dataset()
    #train()
    #create_downsampling_dataset(source_dir, "/home/zju/mfe/dataset/UCMerced_LandUse_downsample_x8/", 1/8)
    #create_downsampling_dataset(source_dir, "/home/zju/mfe/dataset/UCMerced_LandUse_downsample_x4/", 1/4)
    #evaluate("ucmerced", source_dir, 21)

    #evaluate("ucmerced_validate", "/home/zju/mfe/dataset/validate", 21, savepath="/home/zju/mfe/land_cover_classification/confusion_matrix_validate-label.png")
    #evaluate("ucmerced_test", "/home/zju/mfe/dataset/test", 21, savepath="/home/zju/mfe/land_cover_classification/confusion_matrix_test-label.png")
    #evaluate("UCMerced_LandUse_downsample_x4Down", "/home/zju/mfe/dataset/UCMerced_LandUse_downsample_x4/", 21, savepath="/home/zju/mfe/land_cover_classification/confusion_matrix_downsamplex4-label.png")
    #evaluate("UCMerced_LandUse_bicubic_x4SR", "/home/zju/mfe/dataset/UCMerced_LandUse_bicubic_x4/", 21, savepath="/home/zju/mfe/land_cover_classification/confusion_matrix_Bicubicx4SR-label.png")
    #evaluate("UCMerced_LandUse_downsample_x8Down", "/home/zju/mfe/dataset/UCMerced_LandUse_downsample_x8/", 21, savepath="/home/zju/mfe/land_cover_classification/confusion_matrix_downsamplex8-label.png")
    #evaluate("UCMerced_LandUse_bicubic_x8SR", "/home/zju/mfe/dataset/UCMerced_LandUse_bicubic_x8/", 21, savepath="/home/zju/mfe/land_cover_classification/confusion_matrix_Bicubicx8SR-label.png")
    #evaluate("ucmerced_4xDown", "/home/zju/mfe/dataset/x4/", 21, savepath="/home/zju/mfe/land_cover_classification/confusion_matrix_lrx4-label.png")
    #evaluate("ucmerced_8xDown", "/home/zju/mfe/dataset/x8/", 21, savepath="/home/zju/mfe/land_cover_classification/confusion_matrix_lrx8-label.png")

    #evaluate("ucmerced_BICUBIC_4xSR", "/home/zju/mfe/dataset/x4_bicubic/", 21, savepath="/home/zju/mfe/land_cover_classification/confusion_matrix_bicubicx4-label.png")
    #evaluate("ucmerced_BICUBIC_8xSR", "/home/zju/mfe/dataset/x8_bicubic/", 21, savepath="/home/zju/mfe/land_cover_classification/confusion_matrix_bicubicx8-label.png")
    #evaluate("ucmerced_DRCAN_4xSR", "/home/zju/mfe/dataset/DRCAN_x4/", 21, savepath="/home/zju/mfe/land_cover_classification/confusion_matrix_DRCANx4-label.png")
    #evaluate("ucmerced_DRCAN_8xSR", "/home/zju/mfe/dataset/DRCAN_x8/", 21, savepath="/home/zju/mfe/land_cover_classification/confusion_matrix_DRCANx8-label.png")
    #evaluate("ucmerced_DSSR_4xSR", "/home/zju/mfe/dataset/DSSR_x4/", 21, savepath="/home/zju/mfe/land_cover_classification/confusion_matrix_DSSRx4-label.png")
    #evaluate("ucmerced_DSSR_8xSR", "/home/zju/mfe/dataset/DSSR_x8/", 21, savepath="/home/zju/mfe/land_cover_classification/confusion_matrix_DSSRx8-label.png")
    #evaluate("ucmerced_SRAGAN_4xSR", "/home/zju/mfe/dataset/SRAGAN_x4/", 21, savepath="/home/zju/mfe/land_cover_classification/confusion_matrix_SRAGANx4-label.png")
    #evaluate("ucmerced_SRAGAN_8xSR", "/home/zju/mfe/dataset/SRAGAN_x8/", 21, savepath="/home/zju/mfe/land_cover_classification/confusion_matrix_SRAGANx8-label.png")
    #evaluate("ucmerced_NDSRGAN_4xSR", "/home/zju/mfe/dataset/NDSRGAN_x4/", 21, savepath="/home/zju/mfe/land_cover_classification/confusion_matrix_NDSRGANx4-label.png")
    #evaluate("ucmerced_NDSRGAN_8xSR", "/home/zju/mfe/dataset/NDSRGAN_x8/", 21, savepath="/home/zju/mfe/land_cover_classification/confusion_matrix_NDSRGANx8-label.png")
    #evaluate("ucmerced_AMSSRN_4xSR", "/home/zju/mfe/dataset/AMSSRN_x4/", 21, savepath="/home/zju/mfe/land_cover_classification/confusion_matrix_AMSSRNx4-label.png")
    #evaluate("ucmerced_AMSSRN_8xSR", "/home/zju/mfe/dataset/AMSSRN_x8/", 21, savepath="/home/zju/mfe/land_cover_classification/confusion_matrix_AMSSRNx8-label.png")
    #evaluate("ucmerced_SRADSGAN_4xSR", "/home/zju/mfe/dataset/SRADSGAN_x4/", 21, savepath="/home/zju/mfe/land_cover_classification/confusion_matrix_SRADSGANx4-label.png")
    #evaluate("ucmerced_SRADSGAN_8xSR", "/home/zju/mfe/dataset/SRADSGAN_x8/", 21, savepath="/home/zju/mfe/land_cover_classification/confusion_matrix_SRADSGANx8-label.png")
