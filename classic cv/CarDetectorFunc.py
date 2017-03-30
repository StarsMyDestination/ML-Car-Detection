# coding: utf-8
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.feature import hog
# from sklearn.preprocessing import StandardScaler
import time
from scipy.ndimage.measurements import label


# ## Hyperparmeter Config


class Config(object):
    def __init__(self):
        self.cspace = 'YCrCb'  # 'HSV' 'YCrCb'
        self.use_svd = True
        self.svd_K = 75
        self.spatial_size = (32, 32)

        self.hist_bins = 64
        self.hist_range = (0, 256)

        self.orient = 7
        self.pix_per_cell = 8
        self.cell_per_block = 2
        self.hog_channel = 'ALL'  # 'ALL' or 'GRAY'

        self.ystart = 400
        self.ystop = 700
        self.scale_list = [1, 1.5, 2, 2.5]

    def show(self):
        print('color space   : {}'.format(self.cspace))
        print('--- spatial-bining-color ---')
        print('use svd       : {}'.format(self.use_svd))
        print('svd K         : {}'.format(self.svd_K))
        print('spatial size  : {}'.format(self.spatial_size))
        print('--- color-histogram ----')
        print('hist bins     : {}'.format(self.hist_bins))
        print('hist range    : {}'.format(self.hist_range))
        print('--- HOG ---')
        print('orient        : {}'.format(self.orient))
        print('pix per cell  : {}'.format(self.pix_per_cell))
        print('cell per block: {}'.format(self.cell_per_block))
        print('hog channel   : {}'.format(self.hog_channel))
        print('--- Car Detection ---')
        print('ROI ystart    : {}'.format(self.ystart))
        print('ROI ystop     : {}'.format(self.ystop))
        print('window scale  : {}'.format(self.scale_list))


# Define a function to return some characteristics of the dataset
def print_data_info(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    example_img = mpimg.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype
    # Return data_dict
    print('cars: {}, non-cars: {}, total: {}'.format(data_dict["n_cars"],
                                                     data_dict["n_notcars"],
                                                     data_dict['n_cars'] + data_dict['n_notcars']))
    print('image shape: {}, data type: {}'.format(
        data_dict["image_shape"], data_dict["data_type"]))


# Define a function to return HOG features and visualization
def get_hog_features(img, config, vis=False, feature_vec=True):
    orient = config.orient
    pix_per_cell = config.pix_per_cell
    cell_per_block = config.cell_per_block
    if vis is True:
        features, hog_image = hog(img,
                                  orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=True,
                                  feature_vector=feature_vec)
        return features, hog_image
    else:
        features = hog(img,
                       orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=False,
                       feature_vector=feature_vec)
        return features


#  Define spatial bining color extractor(raw pixels, color and shape feature)

def img_svd(X, K=30):
    '''X is a [num_samples, num_features] array'''
    U, _, _ = np.linalg.svd(np.dot(X.T, X))
#     return np.dot(X, U[:, 0:K])
    return U[:, 0:K]


def get_main_feature_vectors(imgs_path, config):
    cspace = config.cspace
    svd_K = config.svd_K
    spatial_size = config.spatial_size
    channel_1, channel_2, channel_3 = [], [], []
    for i, img in enumerate(imgs_path):
        # Read in each one by one
        image = mpimg.imread(img)  # RGB format
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
        else:
            feature_image = np.copy(image)
        feature_image = cv2.resize(feature_image, dsize=spatial_size)
        channel_1.append(feature_image[:, :, 0].ravel())
        channel_2.append(feature_image[:, :, 1].ravel())
        channel_3.append(feature_image[:, :, 2].ravel())
    # a 2D array [num_samples, img_width*img_height]
    channel_1_features = np.stack(channel_1)
    channel_2_features = np.stack(channel_2)
    channel_3_features = np.stack(channel_3)
    print('samples num: {}'.format(len(channel_1)))
    print('image shape: {}'.format(channel_1[0].shape))
    print('feature shape([samples_num, image_size]): {}'.format(
        channel_1_features.shape))

    print('\nCompute each channel main feature vectors...')
    s_time = time.time()
    U1 = img_svd(channel_1_features, K=svd_K)
    U2 = img_svd(channel_2_features, K=svd_K)
    U3 = img_svd(channel_3_features, K=svd_K)
    print('spend {} sec.'.format(time.time() - s_time))
    print('U.shape: {}'.format(U1.shape))

    return U1, U2, U3

# Define a function to compute binned color features


def bin_spatial(img, config, subspace=None):
    # Use cv2.resize().ravel() to create the feature vector
    size = config.spatial_size
    features = cv2.resize(img, size)
    if subspace is None:
        return features.ravel()
    else:
        chan_1 = np.dot(features[:, :, 0].ravel(), subspace[0])
        chan_2 = np.dot(features[:, :, 1].ravel(), subspace[1])
        chan_3 = np.dot(features[:, :, 2].ravel(), subspace[2])
    #     print('chan_3.shape: {}'.format(chan_3.shape))
        # Return the feature vector
        return np.concatenate([chan_1, chan_2, chan_3])


# ### Define Histogram of color (color feature)

# Define a function to compute color histogram features
def color_hist(img, config):
    nbins = config.hist_bins
    bins_range = config.hist_range
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# ### Define global features extractor

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, config, svd_subspace):
    cspace = config.cspace
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for i, img in enumerate(imgs):
        # Read in each one by one
        image = mpimg.imread(img)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
        else:
            feature_image = np.copy(image)
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, config=config, subspace=svd_subspace)
#         spatial_features = np.array([])
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, config)
#         hist_features = np.array([])
        if config.hog_channel == 'GRAY':
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            hog_features = get_hog_features(gray, config=config)
        elif config.hog_channel == 'ALL':
            hog_feat_1 = get_hog_features(feature_image[:, :, 0], config)
            hog_feat_2 = get_hog_features(feature_image[:, :, 1], config)
            hog_feat_3 = get_hog_features(feature_image[:, :, 2], config)
            hog_features = np.hstack((hog_feat_1, hog_feat_2, hog_feat_3))
        if i == 0:
            print('spatial_features.shape: {}'.format(spatial_features.shape))
            print('hist_features.shape: {}'.format(hist_features.shape))
            print('hog_features.shape: {}'.format(hog_features.shape))
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features, hog_features)))
    # Return list of feature vectors
    return features


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def find_cars_multi_scaler(image, config, svd_subspace, svm_model):
    cspace = config.cspace
    ystart = config.ystart
    ystop = config.ystop
    scaler_list = config.scale_list
    svc = svm_model['svc']
    X_scaler = svm_model['X_scaler']

    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
    else:
        feature_image = np.copy(image)

    _m = len(scaler_list)
    hot_windows = []
    for i, scaler in enumerate(scaler_list):
        if i < int(_m / 2):
            _ystart = ystart
            _ystop = int((ystart + ystop) / 2)
        else:
            _ystart = int((ystart + ystop) / 2)
            _ystop = ystop
        hot_windows.extend(find_cars(feature_image, config, svd_subspace, scaler, svc, X_scaler, _ystart, _ystop))


def find_cars(feature_image, config, svd_subspace, scaler, svc, X_scaler, ystart, ystop):
    pix_per_cell = config.pix_per_cell

    img_tosearch = feature_image[ystart:ystop, :, :]
    if scaler != 1:
        imshape = img_tosearch.shape
        img_tosearch = cv2.resize(img_tosearch, (np.int(imshape[1] / scaler), np.int(imshape[0] / scaler)))

    ch1 = img_tosearch[:, :, 0]
    ch2 = img_tosearch[:, :, 1]
    ch3 = img_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, config, feature_vec=False)
    hog2 = get_hog_features(ch2, config, feature_vec=False)
    hog3 = get_hog_features(ch3, config, feature_vec=False)

    hot_windows = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(img_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, config, subspace=svd_subspace)
            hist_features = color_hist(subimg, config)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scaler)
                ytop_draw = np.int(ytop * scaler)
                win_draw = np.int(window * scaler)
                hot_windows.append(((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
    return hot_windows


# # Heatmap

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def get_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    bboxes = []
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bboxes.append(bbox)
    return bboxes


def get_heatmap(img, hot_windows, threshold):
    '''
    return bboxes, heatmap
    '''
    box_list = hot_windows
    # Read in image similar to one shown above
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat, box_list)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, threshold)
    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    bboxes = get_labeled_bboxes(np.copy(img), labels)
    return bboxes, heatmap


def plot_heatmap(img, hot_windows, threshold=3):
    bboxes, heatmap = get_heatmap(img, hot_windows, threshold)
    fig = plt.figure(figsize=(15, 10))
    plt.subplot(121)
    img = draw_boxes(img, bboxes)
    plt.imshow(img)
    plt.title('Car Positions')
    plt.subplot(122)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map')
    fig.tight_layout()
