import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
from model.model import InpaintModel
from tools.mask import get_mask
from sklearn.linear_model import LinearRegression as LR


train_dir = "./data/dataset/train/"
test_dir = "./data/dataset/test/"
truth_dir = "./data/dataset/truth/"
land_path = "./data/dataset/0218.tif"

train_img_paths = glob.glob(train_dir + '*.tif')
test_img_paths = glob.glob(test_dir + '*.tif')
raw_img_paths = glob.glob(truth_dir + '*.tif')

train_img_paths = ('./data/dataset/train/0718.tif', )


def draw_scatter(pairs, axis=0):
    method = 'Biharmonic'
    values_x = []
    values_y = []
    for pair in pairs:
        values_x.append(pair[0][axis])
        values_y.append(pair[1][axis])
    # linear regression
    reg = LR().fit(np.array(values_x).reshape((-1, 1)),
                   np.array(values_y).reshape((-1, 1)))
    x_coor = np.array(range(0, 255, 10)).reshape((-1, 1))
    y_coor = np.array(reg.predict(x_coor)).reshape((-1, 1))
    print('coefficient of axis ', axis, ': ', reg.coef_)
    print('interception of axis ', axis, ': ', reg.intercept_)

    scatter = plt.scatter(values_x, values_y, s=4, c='red')
    line, = plt.plot(x_coor, y_coor, c='blue', linewidth=1.5)
    plt.xlabel('predict value')
    plt.ylabel('truth value')
    plt.legend((scatter, line), ('pixel', 'linear regression'), loc='best')
    if axis == 0:
        title = method + '_H'
        plt.title(title)
    elif axis == 1:
        title = method + '_S'
        plt.title(title)
    elif axis == 2:
        title = method + '_V'
        plt.title(title)
    else:
        raise ValueError('Axis should in (0, 1, 2)')
    plt.show()
    plt.savefig('./results/pred_truth/' + title + '.png')


def main():
    train_imgs = map(cv.imread, train_img_paths)
    test_imgs = map(cv.imread, test_img_paths)

    count = 0
    mse_sum = 0
    model = InpaintModel(land_path)
    for img in train_imgs:
        img_path = train_img_paths[count]
        img_mask = get_mask(img_path)
        # perform inpainting and compute mse
        res = model.inpaint(img, img_mask)
        mse = np.square(np.subtract(img, res)).mean()
        mse_sum += mse
        # save predicted-truth pixels value
        img_name = train_img_paths[count][-8:]
        truth = cv.imread(truth_dir + img_name)
        print(np.max(truth))
        img_mask_raw = get_mask(img_path)
        pred = res[img_mask_raw == 1]
        truth = truth[img_mask_raw == 1]
        pairs = list(zip(pred, truth))
        draw_scatter(pairs, 0)
        draw_scatter(pairs, 1)
        draw_scatter(pairs, 2)

        count += 1
    mse_avg = mse_sum / count


if __name__ == '__main__':
    main()
