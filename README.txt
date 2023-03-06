1. the libraries required to run the project including the full version of each library:
numpy==1.24.1
matplotlib==3.6.2
opencv-python==4.6.0.66
imutils==0.5.4
parameters==0.2.1
sklearn==0.0.post1
scikit-learn==1.2.0
scikit-image==0.19.3
keras==2.11.0

2. how to run each task and where to look for the output file.

Example:
Task 0:
    if the folder "data" is lost you need to run "refactor_data.py". The folder "data" will be created with
    positive and negative examples for faces.

Task 1: 
    script: "RunProject.py"
        -takes the data from "data/train/positive" and "data/train/negative"
        -takes the svm model from "data/train/salveazaFisiere" (train one if not existent)
        -takes test images from "dir_test_examples" (Parameters)
        -output folder for detections, scores, file_names: "solutions_path_t1" (Parameters)

Task 2:
    script: "RunProject_task2.py"
        -takes the data from "data/train/{character}"
        -takes neural network model from "data/train/salveazaFisiere" (train one if not existent)
        -takes test images from "dir_test_examples" (Parameters)
        -takes the detections from task 1 from "solutions_path_t1" (Parameters)
        -classifies the detections and put the solutions in the output folder "solutions_path_t2" (Parameters)
    run "evalueaza_solutie.py" to see the average precision of the models

The folders for "solutions_path_t1" and "solutions_path_t2" must exist!