import math

import numpy as np

greyscale_wt = [0.299, 0.587, 0.114]

def get_greyscale_image(image, colour_wts):
    """
    Gets an image and weights of each colour and returns the image in greyscale
    :param image: The original image
    :param colour_wts: the weights of each colour in rgb (ints > 0)
    :returns: the image in greyscale
    """
    ###Your code here###
    R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]
    return colour_wts[0] * R + colour_wts[1] * G + colour_wts[2] * B

def reshape_bilinear(image, new_shape):
    """
    Resizes an image to new shape using bilinear interpolation method
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :returns: the image resized to new_shape
    """
    in_height, in_width, c = image.shape
    out_height, out_width = new_shape
    new_image = np.zeros((out_height,out_width,c))
    ###Your code here###
    if out_height != 0:
        w_scale_factor = in_width / out_width

    if out_width != 0:
        h_scale_factor = in_height / out_height

    for i in range(out_height):
        for j in range(out_width):
            x = i * h_scale_factor
            y = j * w_scale_factor
            x_down_round = math.floor(x)
            x_up_round = min(in_height - 1, math.ceil(x))
            y_down_round = math.floor(y)
            y_up_round = min(in_width - 1, math.ceil(y))
            if x_up_round == x_down_round and y_up_round == y_down_round:
                c_final = image[int(x), int(y), :]
            elif x_up_round == x_down_round:
                c_1_2 = image[int(x), int(y_down_round), :]
                c_3_4 = image[int(x), int(y_up_round), :]
                c_final = c_1_2 * (y_up_round - y) + c_3_4 * (y - y_down_round)
            elif y_up_round == y_down_round:
                c_1_2 = image[int(x_down_round), int(y), :]
                c_3_4 = image[int(x_up_round), int(y), :]
                c_final = (c_1_2 * (x_up_round - x)) + (c_3_4 * (x - x_down_round))
            else:
                c_1 = image[x_down_round, y_down_round, :]
                c_2 = image[x_up_round, y_down_round, :]
                c_3 = image[x_down_round, y_up_round, :]
                c_4 = image[x_up_round, y_up_round, :]
                c_1_2 = c_1 * (x_up_round - x) + c_2 * (x - x_down_round)
                c_3_4 = c_3 * (x_up_round - x) + c_4 * (x - x_down_round)
                c_final = c_1_2 * (y_up_round - y) + c_3_4 * (y - y_down_round)

            new_image[i, j, :] = c_final

    new_image = new_image.astype(np.uint8)
    return new_image

def gradient_magnitude(image, colour_wts):
    """
    Calculates the gradient image of a given image
    :param image: The original image
    :param colour_wts: the weights of each colour in rgb (> 0)
    :returns: The gradient image
    """
    greyscale = get_greyscale_image(image, colour_wts)
    ###Your code here###
    in_height, in_width = greyscale.shape
    gradient = np.zeros((in_height, in_width,))
    for i in range(in_height):
        for j in range(in_width):
            if i < (in_height - 1):
                diff_x = greyscale[i+1][j] - greyscale[i][j]
            else:
                diff_x = greyscale[0][j] - greyscale[i][j]
            if j < (in_width - 1):
                diff_y = greyscale[i][j+1] - greyscale[i][j]
            else:
                diff_y = greyscale[i][0] - greyscale[i][j]

            gradient[i][j] = math.sqrt((math.pow(diff_x, 2)) + (math.pow(diff_y, 2)))

    return gradient


def visualise_seams(image, new_shape, carving_scheme, colour):
    """
    Visualises the seams that would be removed when reshaping an image to new image (see example in notebook)
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :param carving_scheme: the carving scheme to be used.
    :param colour: the colour of the seams (an array of size 3)
    :returns: an image where the removed seams have been coloured.
    """
    ###Your code here###
    new_height, new_width = new_shape
    if new_width > image.shape[1] or new_width > image.shape[0]:
        print("new shape of the image is larger then the original")
        return image

    copy_origin_image = image.copy()
    num_rotation_height = image.shape[0] - new_height
    num_rotation_width = image.shape[1] - new_width
    if carving_scheme: # 1=true=horizone
        all_sean, new_reduce_picture = findSeam(np.rot90(copy_origin_image), greyscale_wt, num_rotation_height)
        ans = colouring_seams(np.rot90(copy_origin_image),all_sean,colour) #add
        ans = np.rot90(np.rot90(np.rot90(ans)))
    elif carving_scheme == 0: #0=false=vertical
        all_sean, new_reduce_picture = findSeam(copy_origin_image, greyscale_wt, num_rotation_width)
        ans = colouring_seams(copy_origin_image,all_sean,colour)

    return ans

def findSeam(copy_image,greyscale_wt, num_rotation):
    new_indexes_matrix = createCopyOfMatrixWithIndexsInCells(copy_image)
    origin_picture = np.copy(copy_image)
    for i in range(num_rotation):
        e_magnitude = gradient_magnitude(origin_picture, greyscale_wt)
        min_cost_matrix = createCostMatrix(e_magnitude)
        back_tracking_path = createTracking(min_cost_matrix)
        new_indexes_matrix, sean, origin_picture = delete_colm(back_tracking_path, new_indexes_matrix, origin_picture)
        if i == 0:
            all_sean = sean
        else:
            all_sean = np.concatenate((all_sean, sean), axis=0)

    all_sean.astype(int)
    return all_sean, origin_picture

def delete_colm(back_tracking_path, new_indexes_matrix, origin_picture):
    in_height, in_width,c = origin_picture.shape
    deleted_cell = np.zeros((back_tracking_path.shape[0],2),int)
    for i in range(in_height):
        back_tracking_index_to_delete = back_tracking_path.shape[0] - i - 1
        index_to_delete = back_tracking_path[back_tracking_index_to_delete]
        k = np.arange(index_to_delete, in_width, 1, dtype=int)
        m = index_to_delete.astype(np.int64)
        deleted_cell[i] = new_indexes_matrix[i][m]
        for l in k:
            if l != k[k.shape[0]-1]:
                new_indexes_matrix[i][l] = new_indexes_matrix[i][l+1]
                origin_picture[i][l] = origin_picture[i][l+1]

    new_indexes_matrix = new_indexes_matrix[:,:-1]
    origin_picture1 = origin_picture[:,:-1,:]
    return new_indexes_matrix, deleted_cell, origin_picture1

def colouring_seams(image, list_of_seams, colour):
    for seam in list_of_seams:
        image[seam[0], seam[1]] = colour
    return image

def createCopyOfMatrixWithIndexsInCells(image):
    in_height, in_width, c = image.shape
    new_matrix = np.zeros((in_height, in_width,2))
    for i in range(in_height):
        for j in range(in_width):
            new_matrix[i,j] = (i,j)

    return new_matrix

def createTracking(minCostMatrix):
    in_height, in_width = minCostMatrix.shape
    tracking_path_index = np.zeros(in_height)
    for i in reversed(range(in_height)):
        if i == in_height - 1:
            index = np.argmin(minCostMatrix[i, 0:in_width])
        elif index == in_height - 1:
            index += np.argmin(minCostMatrix[i, index - 1:index + 1]) - 1
        elif index == 0:
            index += np.argmin(minCostMatrix[i, index:index + 2])
        else:
            index += np.argmin(minCostMatrix[i, index - 1:index + 2]) - 1

        tracking_path_index[in_height - i - 1] = index

    return tracking_path_index

def createCostMatrix(image):
    in_height, in_width = image.shape
    min_seam_matrix = np.copy(image)
    for i in range(1, in_height):
        for j in range(0, in_width):
            c_l, c_v, c_r = new_gradient_edges(image, i, j)
            if j == 0:
                min_energy = np.min(np.array([min_seam_matrix[i-1,j]+c_v, min_seam_matrix[i-1,j+1]+c_r]))

            elif j == in_width-1:
                min_energy = np.min(np.array([min_seam_matrix[i-1,j]+c_v, min_seam_matrix[i-1,j-1]+c_l]))

            else:
                min_energy = np.min(np.array([min_seam_matrix[i-1,j]+c_v, min_seam_matrix[i-1,j+1]+c_r, min_seam_matrix[i-1,j-1]+c_l]))

            min_seam_matrix[i,j] += min_energy
            min_energy = 0

    return min_seam_matrix


def new_gradient_edges(image, i, j):
    in_height, in_width = image.shape
    if j == 0 or j == in_width - 1:
        c_r = 0
    else:
        c_r = np.abs(image[i, j + 1] - image[i, j - 1])

    if j == in_width - 1:
        c_r = 0
    else:
        c_r = c_r + np.abs(image[i, j + 1] - image[i - 1, j])

    c_l = c_r + np.abs(image[i - 1, j] - image[i, j - 1])

    return c_l, c_r, c_r

def reshape_seam_crarving(image, new_shape, carving_scheme):
    """
    Resizes an image to new shape using seam carving
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :param carving_scheme: the carving scheme to be used.
    :returns: the image resized to new_shape
    """
    ###Your code here###
    new_height, new_width = new_shape

    if new_width > image.shape[1] or new_width > image.shape[0]:
        print("new shape of the image is larger then the original")
        return image

    copy_origin_image = image.copy()
    num_rotation_height = image.shape[0] - new_height
    num_rotation_width = image.shape[1] - new_width
    if carving_scheme:  # 1=true=horizone
        all_sean, new_reduce_picture = findSeam(np.rot90(copy_origin_image), greyscale_wt, num_rotation_height)
        all_sean, new_reduce_picture = findSeam(np.rot90(np.rot90(np.rot90(new_reduce_picture))), greyscale_wt, num_rotation_width)
        new_image = new_reduce_picture
    elif carving_scheme == 0:  # 0=false=vertical
        all_sean, new_reduce_picture = findSeam(copy_origin_image, greyscale_wt, num_rotation_width)
        all_sean, new_reduce_picture = findSeam(np.rot90(new_reduce_picture), greyscale_wt, num_rotation_height)
        new_image = np.rot90(np.rot90(np.rot90(new_reduce_picture)))
    else:
        new_reduce_picture = np.copy(copy_origin_image)
        while i < num_rotation_height or j < num_rotation_width:
            if i < num_rotation_height:
                all_sean, new_reduce_picture = findSeam(np.rot90(new_reduce_picture), greyscale_wt, 1)
                new_reduce_picture = np.rot90(np.rot90(np.rot90(new_reduce_picture)))
                i+=1
            if j < num_rotation_width:
                all_sean, new_reduce_picture = findSeam(new_reduce_picture, greyscale_wt, 1)
                j+=1

    return new_image