import os

class Parameters:
    def __init__(self):
        self.solutions_path_t1 = 'evaluare/fisiere_solutie/352_Burdusa_Petru/task1'
        self.solutions_path_t2 = 'evaluare/fisiere_solutie/352_Burdusa_Petru/task2'
        self.base_dir = 'data/train'
        self.dir_pos_examples = os.path.join(self.base_dir, 'positive')
        self.dir_neg_examples = os.path.join(self.base_dir, 'negative')
        self.dir_test_examples = os.path.join('validare/Validare')
        self.path_annotations = os.path.join('validare/task1_gt_validare.txt')
        self.dir_save_files = os.path.join(self.base_dir, 'salveazaFisiere')
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))

        # set the parameters
        self.dim_window = 72  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
        self.dim_hog_cell = 8  # dimensiunea celulei
        self.dim_descriptor_cell = 72  # dimensiunea descriptorului unei celule
        self.overlap = 0.3
        self.number_positive_examples = 6713  # numarul exemplelor pozitive
        self.number_negative_examples = 30000  # numarul exemplelor negative
        self.overlap = 0.3
        self.has_annotations = True
        self.threshold = 0.0
        self.resize_scale = 0.9
        self.min_width = 75
