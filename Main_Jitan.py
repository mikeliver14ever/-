import tensorflow as tf
from keras_tuner.tuners import BayesianOptimization
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_curve, auc
import seaborn as sns
import pickle

# Define the NeuralNetwork class
class NeuralNetwork(tf.keras.Model):
    # Constructor
    def __init__(self, input_shape, output_shape, learning_rate):
        super(NeuralNetwork, self).__init__()
        # Define layers
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.dense_output = tf.keras.layers.Dense(output_shape, activation='softmax')
        # Define optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Forward pass
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense_output(x)

# Function to train the neural network
def train_neural_network(model, X_train, y_train, epochs=10):
    model.compile(optimizer=model.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, verbose=0)
    return history

# Function to evaluate the neural network
def evaluate_neural_network(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)

# Function to save the model
def save_model(model, model_dir="saved_model"):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save(model_dir)
    print("Model saved successfully.")

# Function to load the model
def load_model(model_dir="saved_model"):
    if os.path.exists(model_dir):
        model = tf.keras.models.load_model(model_dir)
        print("Model loaded successfully.")
        return model
    else:
        print("Model directory not found.")
        return None

# Function to save the TensorFlow Lite model
def save_model_tf_lite(model, model_dir="tf_lite_model"):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    tf_lite_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = tf_lite_converter.convert()
    with open(os.path.join(model_dir, "model.tflite"), "wb") as f:
        f.write(tflite_model)
    print("TensorFlow Lite model saved successfully.")

# Function to plot the learning curve
def plot_learning_curve(history, save_path=None):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Learning Curve')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# Function to save the training history
def save_training_history(history, save_path="training_history.npy"):
    np.save(save_path, history.history)
    print("Training history saved successfully.")

# Function to load the training history
def load_training_history(load_path="training_history.npy"):
    if os.path.exists(load_path):
        history = np.load(load_path, allow_pickle=True).item()
        print("Training history loaded successfully.")
        return history
    else:
        print("Training history file not found.")
        return None

# Function to save the class labels
def save_class_labels(classes, save_path="class_labels.pkl"):
    with open(save_path, 'wb') as f:
        pickle.dump(classes, f)
    print("Class labels saved successfully.")

# Function to load the class labels
def load_class_labels(load_path="class_labels.pkl"):
    with open(load_path, 'rb') as f:
        classes = pickle.load(f)
    print("Class labels loaded successfully.")
    return classes

# Function to save the predictions
def save_predictions(predictions, save_path="predictions.npy"):
    np.save(save_path, predictions)
    print("Predictions saved successfully.")

# Function to load the predictions
def load_predictions(load_path="predictions.npy"):
    if os.path.exists(load_path):
        predictions = np.load(load_path)
        print("Predictions loaded successfully.")
        return predictions
    else:
        print("Predictions file not found.")
        return None

# Function to save the classification report
def save_classification_report(y_true, y_pred, save_path="classification_report.txt"):
    report = classification_report(y_true, y_pred)
    with open(save_path, 'w') as f:
        f.write(report)
    print("Classification report saved successfully.")

# Function to load the classification report
def load_classification_report(load_path="classification_report.txt"):
    if os.path.exists(load_path):
        with open(load_path, 'r') as f:
            report = f.read()
        print("Classification report loaded successfully.")
        return report
    else:
        print("Classification report file not found.")
        return None

# Function to save the ROC curve
def save_roc_curve(y_true, y_pred, save_path="roc_curve.png"):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    print("ROC curve saved successfully.")

# Function to load the ROC curve
def load_roc_curve(load_path="roc_curve.png"):
    if os.path.exists(load_path):
        plt.figure()
        roc_curve_image = plt.imread(load_path)
        plt.imshow(roc_curve_image)
        plt.axis('off')
        plt.show()
        print("ROC curve loaded successfully.")
    else:
        print("ROC curve file not found.")

# Function to plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# Function to evaluate the model
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred,average=None)
    f1 = f1_score(y_true, y_pred, average=None)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

# Dummy data for demonstration
X_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, size=(100,))
X_test = np.random.rand(20, 10)
y_test = np.random.randint(0, 2, size=(20,))
classes = ['Class 0', 'Class 1']  # Dummy class labels

input_shape = X_train.shape[1:]
output_shape = len(set(y_train))

# Bayesian Optimization hyperparameter tuning (dummy function for demonstration)
def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Dense(output_shape, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

tuner = BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='hyperparameters_tuning',
    project_name='neural_network'
)
tuner.search(X_train, y_train, epochs=10, validation_split=0.2)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)

# Train the neural network with the best hyperparameters
history = train_neural_network(model, X_train, y_train)

# Save the training history
save_training_history(history)

# Plot the learning curve
plot_learning_curve(history)

# Save the trained model
save_model(model)

# Save the TensorFlow Lite model
save_model_tf_lite(model)

# Save the class labels
save_class_labels(classes)

# Evaluate the neural network with test data
evaluate_neural_network(model, X_test, y_test)

# Make predictions
predictions = make_predictions(model, X_test)
save_predictions(predictions)

# Visualize the confusion matrix
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(predictions, axis=1)
plot_confusion_matrix(y_true, y_pred, classes)

# Evaluate the model
evaluate_model(y_true, y_pred)

# Save the classification report
save_classification_report(y_true, y_pred)

# Save the ROC curve
save_roc_curve(y_true, y_pred)
