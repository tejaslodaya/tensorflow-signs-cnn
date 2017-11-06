import numpy as np
from scipy import ndimage
from cnn_utils import load_dataset, preprocess_data
from tf_utils import model
import scipy

num_classes = 6

# Loading the data (signs)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Preprocess data
X_train, X_test, Y_train, Y_test = preprocess_data(X_train_orig, X_test_orig, Y_train_orig, Y_test_orig, num_classes)

_, _, parameters = model(X_train, Y_train, X_test, Y_test, num_epochs=6)

#TODO: Use parameters and use it in predict workflow
fname = "images/thumbs_up.jpg"
image = np.array(ndimage.imread(fname, flatten=False))
prediction_image = scipy.misc.imresize(image, size=(64,64))



