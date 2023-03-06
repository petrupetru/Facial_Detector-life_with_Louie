from FacialDetector import *
from Parameters import *
import numpy as np
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
import imutils

solution_path_task1 = "evaluare/fisiere_solutie/331_Alexe_Bogdan/task1/"
solution_path_task2 = "evaluare/fisiere_solutie/331_Alexe_Bogdan/task2/"
def evaluate_results_task1(solution_path):
    # incarca detectiile + scorurile + numele de imagini
    detections = np.load(solution_path + "detections_all_faces.npy", allow_pickle=True, fix_imports=True,
                         encoding='latin1')
    print(detections.shape)

    scores = np.load(solution_path + "scores_all_faces.npy", allow_pickle=True, fix_imports=True, encoding='latin1')
    print(scores.shape)

    file_names = np.load(solution_path + "file_names_all_faces.npy", allow_pickle=True, fix_imports=True,
                         encoding='latin1')
    print(file_names.shape)

def evaluate_results_task2(solution_path):
    # incarca detectiile + scorurile + numele de imagini
    detections = np.load(solution_path + "detections_ora.npy", allow_pickle=True, fix_imports=True,
                         encoding='latin1')
    print(detections.shape)

    scores = np.load(solution_path + "scores_ora.npy", allow_pickle=True, fix_imports=True, encoding='latin1')
    print(scores.shape)

    file_names = np.load(solution_path + "file_names_ora.npy", allow_pickle=True, fix_imports=True,
                         encoding='latin1')
    print(file_names.shape)



evaluate_results_task1(solution_path_task1)
evaluate_results_task2(solution_path_task2)

