from Parameters import *
import numpy as np
from numpy import asarray
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import pdb
import pickle
import ntpath
from copy import deepcopy
import timeit
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from skimage import data, exposure
import imutils


solution_path_task1 = "evaluare/fisiere_solutie/352_Burdusa_Petru/task1/"
path_faces = 'data/train/'
andy = 'andy'
louie = 'louie'
ora = 'ora'
tommy = 'tommy'
unknown = 'unknown'
andy_label = 0
louie_label = 1
ora_label = 2
tommy_label = 3
params: Parameters = Parameters()


def load_detections_task1(solution_path):
    # incarca detectiile + scorurile + numele de imagini
    detections = np.load(solution_path + "detections_all_faces.npy", allow_pickle=True, fix_imports=True,
                         encoding='latin1')
    scores = np.load(solution_path + "scores_all_faces.npy", allow_pickle=True, fix_imports=True, encoding='latin1')

    file_names = np.load(solution_path + "file_names_all_faces.npy", allow_pickle=True, fix_imports=True,
                         encoding='latin1')
    return detections,scores,file_names


def load_train_images(path):
    """get all jpg files from path"""
    images = []
    names = []
    files = os.listdir(path)
    for file in files:
        if file[-3:] == 'jpg':
            img = cv.imread(path + '/' + file, cv.IMREAD_GRAYSCALE)
            img_flip = cv.flip(img, 1)
            img = hog(img, pixels_per_cell=(params.dim_hog_cell, params.dim_hog_cell),
                      cells_per_block=(2, 2), feature_vector=True)
            img_flip = hog(img_flip, pixels_per_cell=(params.dim_hog_cell, params.dim_hog_cell),
                      cells_per_block=(2, 2), feature_vector=True)
            img = asarray(img).flatten()
            img_flip = asarray(img_flip).flatten()
            # img = cv.resize(img, (0, 0), fx=resize_width, fy=resize_height)
            images.append(img)
            images.append(img_flip)
            names.append(path[10] + file)
    return images, names

def load_images(path):
    """get all jpg files from path"""
    images = {}
    files = os.listdir(path)
    for file in files:
        if file[-3:] == 'jpg':
            img = cv.imread(path + '/' + file, cv.IMREAD_GRAYSCALE)
            #img = cv.resize(img, (0, 0), fx=resize_width, fy=resize_height)
            images[file] = img
    return images

def train_model(x_train, y_train):
    learning_rate = 0.0007
    alpha = 0.0000085
    hidden_layer_sizes = (128, 64, 64, 64, 64, 64, 64, 64, 32, 32, 32, 32, 32, 32, 16, 16, 16, 16)
    model_path = "data/train/salveazaFisiere/"
    model_file_name = f"NN_model_{learning_rate}_{alpha}_{len(hidden_layer_sizes)}"
    if os.path.exists(model_path + model_file_name):
        print("found model")
        model = pickle.load(open(model_path + model_file_name, 'rb'))
    else:
        print("model not found! training one")
        model = MLPClassifier(learning_rate_init=learning_rate, solver='adam', alpha=0.00001, hidden_layer_sizes=hidden_layer_sizes, verbose=True, max_iter=500, random_state=1)
        model.fit(x_train, y_train)
        pickle.dump(model, open(model_path + model_file_name, 'wb'))
    return model

def test_model(model, detections, scores, file_names, test_images):
    test_data = []
    for (detection, file_name) in zip(detections, file_names):
        image = test_images[file_name]
        patch = image[int(detection[1]) : int(detection[3]), int(detection[0]):int(detection[2])]
        patch = cv.resize(patch, (72, 72))
        patch = hog(patch, pixels_per_cell=(params.dim_hog_cell, params.dim_hog_cell),
                  cells_per_block=(2, 2), feature_vector=True)
        patch = asarray(patch).flatten()
        test_data.append(patch)
    test_data = asarray(test_data)
    predictions = model.predict(test_data)
    print("$$$$")
    for p in predictions:
        print(p)
    andy_detections = []
    andy_scores = []
    andy_files = []
    louie_detections = []
    louie_scores = []
    louie_files = []
    ora_detections = []
    ora_scores = []
    ora_files = []
    tommy_detections = []
    tommy_scores = []
    tommy_files = []
    for i in range(len(predictions)):
        if predictions[i] == andy_label:
            andy_detections.append(detections[i])
            andy_scores.append(scores[i])
            andy_files.append(file_names[i])
        if predictions[i] == louie_label:
            louie_detections.append(detections[i])
            louie_scores.append(scores[i])
            louie_files.append(file_names[i])
        if predictions[i] == ora_label:
            ora_detections.append(detections[i])
            ora_scores.append(scores[i])
            ora_files.append(file_names[i])
        if predictions[i] == tommy_label:
            tommy_detections.append(detections[i])
            tommy_scores.append(scores[i])
            tommy_files.append(file_names[i])
    andy_detections = asarray(andy_detections)
    andy_scores = asarray(andy_scores)
    andy_files = asarray(andy_files)
    louie_detections = asarray(louie_detections)
    louie_scores = asarray(louie_scores)
    louie_files = asarray(louie_files)
    ora_detections = asarray(ora_detections)
    ora_scores = asarray(ora_scores)
    ora_files = asarray(ora_files)
    tommy_detections = asarray(tommy_detections)
    tommy_scores = asarray(tommy_scores)
    tommy_files = asarray(tommy_files)

    np.save(params.solutions_path_t2 + "/detections_andy.npy", arr=andy_detections, allow_pickle=True,fix_imports=True)
    np.save(params.solutions_path_t2 + "/scores_andy.npy", arr=andy_scores, allow_pickle=True, fix_imports=True)
    np.save(params.solutions_path_t2 + "/file_names_andy.npy", arr=andy_files, allow_pickle=True,fix_imports=True)

    np.save(params.solutions_path_t2 + "/detections_louie.npy", arr=louie_detections, allow_pickle=True,fix_imports=True)
    np.save(params.solutions_path_t2 + "/scores_louie.npy", arr=louie_scores, allow_pickle=True, fix_imports=True)
    np.save(params.solutions_path_t2 + "/file_names_louie.npy", arr=louie_files, allow_pickle=True,fix_imports=True)

    np.save(params.solutions_path_t2 + "/detections_ora.npy", arr=ora_detections, allow_pickle=True,fix_imports=True)
    np.save(params.solutions_path_t2 + "/scores_ora.npy", arr=ora_scores, allow_pickle=True, fix_imports=True)
    np.save(params.solutions_path_t2 + "/file_names_ora.npy", arr=ora_files, allow_pickle=True,fix_imports=True)

    np.save(params.solutions_path_t2 + "/detections_tommy.npy", arr=tommy_detections, allow_pickle=True,fix_imports=True)
    np.save(params.solutions_path_t2 + "/scores_tommy.npy", arr=tommy_scores, allow_pickle=True, fix_imports=True)
    np.save(params.solutions_path_t2 + "/file_names_tommy.npy", arr=tommy_files, allow_pickle=True,fix_imports=True)

if __name__ == '__main__':
    detections_t1, scores_t1, file_names_t1 = load_detections_task1(solution_path_task1)
    test_images = load_images(params.dir_test_examples)
    print(type(test_images))

    andy_faces, andy_file_names = load_train_images(path_faces + andy)
    louie_faces, louie_file_names = load_train_images(path_faces + louie)
    ora_faces, ora_file_names = load_train_images(path_faces + ora)
    tommy_faces, tommy_file_names = load_train_images(path_faces + tommy)
    faces = andy_faces + louie_faces + ora_faces + tommy_faces
    labels = ([andy_label for _ in range(len(andy_faces))]
            + [louie_label for _ in range(len(louie_faces))]
            + [ora_label for _ in range(len(ora_faces))]
            + [tommy_label for _ in range(len(tommy_faces))])
    train_data = asarray(faces)
    train_labels = asarray(labels)
    print(train_data.shape)


    model = train_model(train_data, train_labels)
    test_model(model, detections_t1, scores_t1, file_names_t1, test_images)


