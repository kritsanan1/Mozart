import cv2
import random
import imutils
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import pickle

dataset_path = 'train_data/data'
target_img_size = (100, 100)
sample_count = 50


def extract_raw_pixels(img):
    resized = cv2.resize(img, target_img_size)
    return resized.flatten()


def extract_hsv_histogram(img):
    resized = cv2.resize(img, target_img_size)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    if imutils.is_cv2():
        hist = cv2.normalize(hist)
    else:
        cv2.normalize(hist, hist)
    return hist.flatten()


def extract_hog_features(img):
    img = cv2.resize(img, target_img_size)
    win_size = (100, 100)
    cell_size = (5, 5)  # Smaller cells for more detail
    block_size_in_cells = (3, 3)  # Larger blocks for better context

    block_size = (block_size_in_cells[1] * cell_size[1],
                  block_size_in_cells[0] * cell_size[0])
    block_stride = (cell_size[1], cell_size[0])
    nbins = 12  # More orientation bins for better discrimination
    hog = cv2.HOGDescriptor(win_size, block_size,
                            block_stride, cell_size, nbins)
    h = hog.compute(img)
    return h.flatten()


def extract_hog_features_enhanced(img):
    """Enhanced HOG feature extraction with optimized parameters for musical symbols"""
    img = cv2.resize(img, target_img_size)
    win_size = (100, 100)
    cell_size = (5, 5)  # Smaller cells for capturing fine details
    block_size_in_cells = (3, 3)  # Larger blocks for better spatial context

    block_size = (block_size_in_cells[1] * cell_size[1],
                  block_size_in_cells[0] * cell_size[0])
    block_stride = (cell_size[1], cell_size[0])
    nbins = 12  # More orientation bins for better feature discrimination
    
    # Create HOG descriptor with enhanced parameters
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    h = hog.compute(img)
    
    # Handle case where HOG returns None
    if h is None:
        return np.zeros((win_size[0] // cell_size[0] * win_size[1] // cell_size[1] * nbins * block_size_in_cells[0] * block_size_in_cells[1],))
    
    h = h.flatten()
    
    # Normalize features to prevent dominance by lighting variations
    if len(h) > 0:
        norm = np.linalg.norm(h)
        if norm > 0:
            h = h / norm
    
    return h


def extract_features(img, feature_set='raw'):
    if feature_set == 'hog':
        return extract_hog_features_enhanced(img)
    elif feature_set == 'hog_original':
        return extract_hog_features(img)
    elif feature_set == 'raw':
        return extract_raw_pixels(img)
    else:
        return extract_hsv_histogram(img)


def load_dataset(feature_set='raw', dir_names=[]):
    features = []
    labels = []
    count = 0
    for dir_name in dir_names:
        print(dir_name)
        imgs = glob(f'{dataset_path}/{dir_name}/*.png')
        count += len(imgs)
        subset = random.sample([i for i in range(len(imgs))], min(len(imgs), sample_count))
        for i in subset:
            img = cv2.imread(imgs[i])
            labels.append(dir_name)
            features.append(extract_features(img, feature_set))
    print(f'Total: {len(dir_names)} directories, and {count} images')
    return features, labels


def load_classifiers():
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)

    classifiers = {
        'SVM': svm.LinearSVC(random_state=random_seed),
        'KNN': KNeighborsClassifier(n_neighbors=7),
        'NN': MLPClassifier(
            activation='relu', 
            hidden_layer_sizes=(300, 150),  # Deeper network for better learning
            max_iter=20000,                  # More iterations for convergence
            alpha=1e-5,                      # L2 regularization to prevent overfitting
            solver='adam',                     # Adam optimizer for better performance
            verbose=10,
            tol=1e-6,                       # Lower tolerance for better accuracy
            random_state=random_seed,
            learning_rate_init=0.001,         # Higher initial learning rate
            learning_rate='adaptive',        # Adaptive learning rate
            early_stopping=False,              # Disable early stopping to avoid the issue
            validation_fraction=0.1,         # 10% for validation
            n_iter_no_change=50              # Stop if no improvement for 50 iterations
        )
    }
    return classifiers, random_seed


def run_experiment(classifier='SVM', feature_set='hog', dir_names=[]):
    from sklearn.preprocessing import LabelEncoder
    
    print('Loading dataset. This will take time ...')
    features, labels = load_dataset(feature_set, dir_names)
    print('Finished loading dataset.')

    # Convert to numpy arrays and ensure proper data types
    features = np.array(features, dtype=np.float64)
    
    # Encode labels to ensure they're numeric
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    
    classifiers, random_seed = load_classifiers()

    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels_encoded, test_size=0.2, random_state=random_seed)

    model = classifiers[classifier]
    print('############## Training', classifier, "##############")
    model.fit(train_features, train_labels)
    accuracy = model.score(test_features, test_labels)
    print(classifier, 'accuracy:', accuracy*100, '%')

    return model, accuracy


def train(model_name, feature_name, saved_model_name):
    dir_names = [path.split('/')[2] for path in glob(f'{dataset_path}/*')]

    model, accuracy = run_experiment(model_name, feature_name, dir_names)

    filename = f'trained_models/{saved_model_name}.sav'
    pickle.dump(model, open(filename, 'wb'))


if __name__ == "__main__":
    # Train multiple models for comparison
    print("Training enhanced neural network with HOG features...")
    train('NN', 'hog', 'nn_trained_model_hog_enhanced')
    
    print("Training SVM with HOG features...")
    train('SVM', 'hog', 'svm_trained_model_hog')
    
    print("Training KNN with HOG features...")
    train('KNN', 'hog', 'knn_trained_model_hog')
