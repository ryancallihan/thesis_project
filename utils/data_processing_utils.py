import numpy as np
import logging
import math


def pad_matrix(matrix, img_size):
    new_matrix = np.zeros(img_size).astype('float32')
    if matrix.shape[0] == 0:
        return new_matrix
    elif len(matrix.shape) < 2:
        x_size = matrix.shape[0]
        y_size = 1
    else:
        x_size, y_size = matrix.shape

    if x_size >= img_size[0]:
        x_size = img_size[0]
    if y_size >= img_size[1]:
        y_size = img_size[1]
    new_matrix[:x_size, :y_size] = matrix[:x_size, :y_size]
    return new_matrix


def split_matrices(features, labels, max_frames, input='3d'):
    print('features', len(features), 'labels', len(features))
    new_features = []
    new_labels = []
    shapes = []
    for i, m in enumerate(features):

        if i % 500 == 0:
            logging.info('Splitting %d of %d' % (i, len(features)))
        # if m.shape[0] < max_frames and len(m) > 64:
        #     shapes.append(list(m.shape))
        #     matrix = np.zeros((max_frames, m.shape[1]))
        #     matrix[:m.shape[0], :m.shape[1]] = m
        #     new_features.append(matrix)
        #     new_labels.append(labels[i])
        # else:
        divide_into = int(math.ceil(m.shape[0] / max_frames))
        for d in range(divide_into):
            if input == 'append':
                matrix = np.zeros((max_frames, m.shape[1]))
            else:
                matrix = np.zeros((max_frames, m.shape[1], m.shape[2]))
            s_idx = d * max_frames
            e_idx = d * max_frames + max_frames
            m_slice = m[s_idx:e_idx, :]
            if len(m_slice) > 64:
                # print('m slice', m_slice.shape)
                shapes.append(list(m_slice.shape))
                # if samples2seconds(len(sig), sample_rate) < 2:
                #     continue
                if input == 'append':
                    matrix[:m_slice.shape[0], :m_slice.shape[1]] = m_slice
                else:
                    matrix[:m_slice.shape[0], :m_slice.shape[1], :] = m_slice
                new_features.append(matrix)
                new_labels.append(labels[i])
    return np.array(new_features), np.array(new_labels), np.array(shapes)
