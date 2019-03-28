import numpy as np
from sklearn import mixture
from PIL import Image
import cv2
import igraph


def calculate_gradients(img, height, width):
    """Find the gradient magnitude in different directions."""
    X = np.sum((img[1:height, 0:width]-img[0:height-1, 0:width])**2, axis=2)
    Y = np.sum((img[0:height, 1:width]-img[0:height, 0:width-1])**2, axis=2)
    XY = np.sum((img[1:height, 1:width]-img[0:height-1, 0:width-1])**2, axis=2)
    YX = np.sum((img[1:height, 0:width-1]-img[0:height-1, 1:width])**2, axis=2)
    return X, Y, XY, YX


def calculate_beta(X, Y, XY, YX, height, width):
    sum_grads = np.sum(X)+np.sum(Y)+np.sum(YX)+np.sum(XY)
    norm_factor = 4*height*width - 3*height - 3*width + 2
    beta = 2*sum_grads/norm_factor
    return 1.0/beta


def get_indices(height, width, direction):
    if direction == "X":
        s = 1, height, 0, width
        t = 0, height-1, 0, width
    if direction == "Y":
        s = 0, height, 1, width
        t = 0, height, 0, width-1
    if direction == "XY":
        s = 1, height, 1, width
        t = 0, height-1, 0, width-1
    if direction == "YX":
        s = 1, height, 0, width-1
        t = 0, height-1, 1, width
    return s, t


def create_edges(height, width, gamma, beta, A, direction):
    s, t = get_indices(height, width, direction)
    sr_0, sr_n, sc_0, sc_n = s
    tr_0, tr_n, tc_0, tc_n = t
    source = np.zeros([height, width])
    source[sr_0:sr_n, sc_0:sc_n] = 1
    source = source.reshape([height*width])
    target = np.zeros([height, width])
    target[tr_0:tr_n, tc_0: tc_n] = 1
    target = target.reshape([height * width])

    edges = np.column_stack([np.where(source == 1)[0], np.where(target == 1)[0]])
    w = gamma*np.exp(-beta*A)
    return edges, w


def GrabCut(img, mask, iter, size):
    """Main GrabCut algo"""
    height, width = size
    gamma = 50

    print("Finding gradients in 4 directions")
    X, Y, XY, YX = calculate_gradients(img, height, width)
    beta = calculate_beta(X, Y, XY, YX, height, width)

    # define edges
    print("Creating edges and weights")
    X = X.reshape([X.shape[0] * X.shape[1]])
    Y = Y.reshape([Y.shape[0] * Y.shape[1]])
    XY = XY.reshape([XY.shape[0] * XY.shape[1]])
    YX = YX.reshape([YX.shape[0] * YX.shape[1]])

    edges, w = create_edges(height, width, gamma, beta, X, direction="X")

    edges_t, w_t = create_edges(height, width, gamma, beta, Y, direction="Y")
    edges = np.row_stack([edges, edges_t])
    w = np.append(w, w_t)

    edges_t, w_t = create_edges(height, width, gamma, beta, XY, direction="XY")
    edges = np.row_stack([edges, edges_t])
    w = np.append(w, w_t)

    edges_t, w_t = create_edges(height, width, gamma, beta, YX, direction="YX")
    edges = np.row_stack([edges, edges_t])
    w = np.append(w, w_t)

    print("Making copies of mask and image")
    K_temp = np.zeros([width*height])
    T_temp = mask.reshape([width*height])
    img_temp = img.reshape([width * height, 3])

    print("Defining Tb, Tf, Tu sets")
    # Define Tb Tf Tu
    Tb = T_temp == 0
    Tf = T_temp == 3
    Tu = (T_temp == 1) | (T_temp == 2)

    print("Initalize GMM")
    gmm = [mixture.GaussianMixture(n_components=5),
           mixture.GaussianMixture(n_components=5)]

    # Edges for Tb and Tf
    if any(Tb):
        edges_t = np.column_stack([np.where(Tb)[0], (width * height) * np.ones(Tb[Tb].shape)])
        edges = np.row_stack([edges, edges_t])
        w_t = 9 * gamma * np.ones(Tb[Tb].shape)
        w = np.append(w, w_t)

        edges_t = np.column_stack([np.where(Tb)[0], (width * height + 1) * np.ones(Tb[Tb].shape)])
        edges = np.row_stack([edges, edges_t])
        w_t = np.zeros(Tb[Tb].shape)
        w = np.append(w, w_t)
    if any(Tf):
        edges_t = np.column_stack([np.where(Tf)[0], (width * height + 1) * np.ones(Tf[Tf].shape)])
        edges = np.row_stack([edges, edges_t])
        w_t = 9 * gamma * np.ones(Tf[Tf].shape)
        w = np.append(w, w_t)

        edges_t = np.column_stack([np.where(Tf)[0], (width * height) * np.ones(Tf[Tf].shape)])
        edges = np.row_stack([edges, edges_t])
        w_t = np.zeros(Tf[Tf].shape)
        w = np.append(w, w_t)

    length = len(w)

    for i in range(iter):
        print("Mincut iteration ", i)
        edges = edges[range(length)]
        w = w[range(length)]

        Tb = (T_temp == 0) | (T_temp == 1)
        Tf = (T_temp == 2) | (T_temp == 3)

        K_temp[Tb] = gmm[0].fit_predict(img_temp[Tb])
        K_temp[Tf] = gmm[1].fit_predict(img_temp[Tf])

        edges_t = np.column_stack([np.where(Tu)[0], (width*height) * np.ones(Tu[Tu].shape)])
        edges = np.row_stack([edges, edges_t])
        w_t = -gmm[1].score(img_temp[Tu])
        w = np.append(w, w_t)

        edges_t = np.column_stack([np.where(Tu)[0], (width*height+1) * np.ones(Tu[Tu].shape)])
        edges = np.row_stack([edges, edges_t])
        w_t = -gmm[0].score(img_temp[Tu])
        w = np.append(w, w_t)

        # Construct the graph
        edges = edges.astype(int)
        cuts = igraph.Graph()
        cuts.es['weight'] = 1
        cuts.add_vertices(height * width + 2)
        cuts.add_edges(edges)
        cuts.es['weight'] = w
        # Mincut
        c = cuts.mincut(width * height, width * height+1, capacity='weight')

        indexb = np.array(list(c[0]), dtype=int)
        indexb = indexb[indexb < width*height]
        indexf = np.array(list(c[1]), dtype=int)
        indexf = indexf[indexf < width*height]

        T_temp[indexb] = 1
        T_temp[indexf] = 2

    img_temp[T_temp < 2] = (255, 255, 255)
    img = img_temp.reshape([height, width, 3])
    img = img.astype(int)

    # Output image
    result = Image.new("RGB", (width, height))
    for i in range(height):
        for j in range(width):
            result.putpixel([j, i], tuple(img[i, j]))
    result.save("out.png")


def minx(x1, x2, y1, y2):
    if x1 < x2:
        return x1, y1
    else:
        return x2, y2


def checkout(x, p):
    return max(0, min(x, p))


def process_img(filename, p1_x, p1_y, p2_x, p2_y, foreground=[], background=[], times=5):
    img = cv2.imread(filename)
    mask = np.zeros(img.shape[:2], np.uint8)
    height = img.shape[0]
    width = img.shape[1]

    p1_x = checkout(p1_x, width-1)
    p1_y = checkout(p1_y, height-1)
    p2_x = checkout(p2_x, width-1)
    p2_y = checkout(p2_y, height-1)
    mask[min(p1_y, p2_y):max(p1_y, p2_y)+1, min(p1_x, p2_x):max(p1_x, p2_x)+1] = 3

    for y1, x1, y2, x2 in foreground:
        x1 = checkout(x1, height-1)
        y1 = checkout(y1, width-1)
        x2 = checkout(x2, height-1)
        y2 = checkout(y2, width-1)
        if x1 == x2:
            mask[x1, min(y1, y2):max(y1, y2)+1] = 1
        else:
            k = (y1-y2)/(x1-x2)
            x, y = minx(x1, x2, y1, y2)
            while True:
                mask[x, y] = 1
                x = x+1
                y = checkout(int(round(y+k)), width-1)
                if x > max(x1, x2):
                    break
    for y1, x1, y2, x2 in background:
        x1 = checkout(x1, height-1)
        y1 = checkout(y1, width-1)
        x2 = checkout(x2, height-1)
        y2 = checkout(y2, width-1)
        if x1 == x2:
            mask[x1, min(y1, y2):max(y1, y2)+1] = 0
        else:
            k = (y1-y2)/(x1-x2)
            x, y = minx(x1, x2, y1, y2)
            while True:
                mask[x, y] = 0
                x = x+1
                y = checkout(int(round(y+k)), height-1)
                if x > max(x1, x2):
                    break

    mask_ = np.zeros(img.shape[:2], np.uint8)
    mask_[mask == 1] = 3
    mask_[mask == 2] = 1
    mask_[mask == 3] = 2
    img = Image.open(filename)
    # transform image to RGB matrix
    img = img.convert('RGB')
    img = np.array(img, dtype=float)
    GrabCut(img, mask_, 1, img.shape[0:2])
    return True
