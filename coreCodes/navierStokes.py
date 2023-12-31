import cv2
import numpy as np

PIXEL_WHITE = 1
PIXEL_RED = 2


class Coord:
    def __init__(self, i, j, color):
        self.i = i
        self.j = j
        self.color = color


def create_mask(mask):
    mask_data = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            color = mask[i, j]
            mask_data.append(Coord(i, j, color))
    return np.array(mask_data)


def perform_bertalmio_pde_inpainting_0(input_array, mask_array, total_iters, total_inpaint_iters,
                                       total_anidiffuse_iters, total_stages, delta_ts, sensitivities, diffuse_coef):
    # initialize output
    output_array = input_array
    # size mask
    size_mask = mask_array.shape[0]
    # compute bertalmio for each stage
    for stage in range(0, total_stages):

        # grab data
        total_iter = total_iters[stage]
        total_inpaint_iter = total_inpaint_iters[stage]
        total_anidiffuse_iter = total_anidiffuse_iters[stage]
        sensitivity = sensitivities[stage]
        delta_t = delta_ts[stage]

        # declare variables
        image_grad_row = cv2.Mat
        image_grad_col = cv2.Mat
        image_grad_norm = cv2.Mat
        image_iso_row = cv2.Mat(input_array)
        image_iso_col = cv2.Mat(input_array)
        image_iso_norm = cv2.Mat(input_array)
        image_laplacian = cv2.Mat(input_array)
        image_laplacian_grad_row = cv2.Mat(input_array)
        image_laplacian_grad_col = cv2.Mat(input_array)
        diffuse_coefs = cv2.Mat
        temp = cv2.Mat

        # run stage of algorithm
        for it in range(0, total_iter):
            for it_aniffuse in range(0, total_anidiffuse_iter):
                image_grad_row = cv2.Sobel(output_array, -1, 0, 1)
                image_grad_col = cv2.Sobel(output_array, -1, 1, 0)
                image_grad_norm = cv2.magnitude(image_grad_row, image_grad_col)
                if diffuse_coef == 0:
                    diffuse_coefs = cv2.exp(-image_grad_norm.mul(1 / sensitivity))
                else:
                    temp = cv2.pow(image_grad_norm * (1 / sensitivity), 2)
                    diffuse_coefs = 1 / (1 + temp)
                output_array = cv2.Laplacian(image_laplacian, -1)

                for cont in range(0, size_mask):
                    coord = mask_array[cont]
                    row = coord.i
                    col = coord.j

                    output_array[row, col] +=\
                        delta_t * diffuse_coefs[row, col] * image_laplacian[row, col]

            # perform inpainting
            for total_anidiffuse_iters in range(total_inpaint_iter):
                output_array = cv2.Sobel(image_iso_row, -1, 1, 0)
                output_array = cv2.Sobel(image_iso_col, -1, 0, 1)
                image_iso_row *= -1
                image_iso_norm = cv2.sqrt(cv2.multiply(image_iso_row, image_iso_row)
                                          + cv2.multiply(image_iso_col, image_iso_col))
                output_array = cv2.Laplacian(image_laplacian, -1)
                image_laplacian = cv2.Sobel(image_laplacian_grad_row, -1, 0, 1)
                image_laplacian = cv2.Sobel(image_laplacian_grad_col, -1, 1, 0)

                for cont in range(0, size_mask):
                    ccord = mask_array[cont]
                    row = ccord.i
                    col = ccord.j
                    if image_iso_norm[row, col] != 0:
                        output_array[row, col] -= delta_t * (
                            image_iso_row[row, col] * image_laplacian_grad_row[row, col] +
                            image_iso_col[row, col] * image_laplacian_grad_col[row, col] /
                            image_iso_norm[row, col]
                        )
                        output_array[row, col] = 1 if output_array[row, col] > 1.0 else output_array[row, col]
                        output_array[row, col] = 0 if output_array[row, col] < 0.0 else output_array[row, col]
    return output_array


def imshowOutput(output_window_name, output_array):
    cv2.imshow(output_window_name, output_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("show time")
    dir = "../data/"
    name = "0718_"
    image_name = dir + name + "test.tif"
    mask_name = dir + name + "mask.tif"
    window_name = image_name
    mask_window_name = mask_name
    output_window_name = "output_array"
    output_array = np.zeros((0, 0))

    image_array = cv2.imread(image_name)
    image_array = image_array.astype(np.float32)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    image_array = cv2.normalize(image_array, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32FC1)

    mask_array = cv2.imread(mask_name)
    mask_data = create_mask(mask_array)

    # Bertalmio PDE Inpainting
    total_iters = np.array([500, ])
    total_inpaint_iters = np.array([6, ])
    total_andiffuse_iters = np.array([6, ])
    total_stages = 1
    delta_ts = np.array([0.02, ])
    sensitivites = np.array([100, ])
    diffuse_coef = np.array([1, ])

    output_array = perform_bertalmio_pde_inpainting_0(image_array, mask_data, total_iters,
                                       total_inpaint_iters, total_andiffuse_iters, total_stages,
                                       delta_ts, sensitivites, diffuse_coef)

    print('alg completed')
    cv2.imwrite('../data/0718_NSRes.tif', output_array)
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(mask_window_name, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(output_window_name, cv2.WINDOW_AUTOSIZE)

    cv2.imshow(window_name, image_array)
    cv2.imshow(mask_window_name, mask_array)
    cv2.imshow(output_window_name, output_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
