from cnn_utils import load_dataset, preprocess_data
from tf_utils import model

num_classes = 6

# Loading the data (signs)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Preprocess data
X_train, X_test, Y_train, Y_test = preprocess_data(X_train_orig, X_test_orig, Y_train_orig, Y_test_orig, num_classes)

# Training the parameters
_, _, parameters = model(X_train, Y_train, X_test, Y_test)