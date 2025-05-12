# Import libraries for both models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml, load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

def logistic_regression_mnist():
    print("\n=== Logistic Regression on MNIST Dataset ===")
    # Load MNIST dataset from OpenML
    mnist = fetch_openml('mnist_784', version=1)

    # Data preprocessing
    X = mnist['data'] / 255.0
    y = mnist['target'].astype(int)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train logistic regression model
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Accuracy: {accuracy * 100:.2f}%")

    # Show predictions for first 5 test images
    for i in range(5):
        plt.imshow(X_test.iloc[i].values.reshape(28, 28), cmap=plt.cm.binary)
        plt.title(f"Predicted: {y_pred[i]}, Actual: {y_test.iloc[i]}")
        plt.axis('off')
        plt.show()

def neural_network_digits():
    print("\n=== Neural Network on Digits Dataset ===")
    digits = load_digits()
    X, y = digits.data, digits.target

    print(f"Total samples: {len(X)}")
    print(f"Image shape: {digits.images[0].shape}")
    print(f"Number of classes: {len(np.unique(y))}")

    # Visualize first 10 digits
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(digits.images[i], cmap='gray')
        plt.title(f'Digit: {digits.target[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('sample_digits.png')
    plt.close()

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # MLP model
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, activation='relu', solver='adam', random_state=42)
    mlp.fit(X_train_scaled, y_train)

    # Evaluation
    y_pred = mlp.predict(X_test_scaled)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Show predictions for first 10 images
    plt.figure(figsize=(12, 6))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(X_test.reshape(-1, 8, 8)[i], cmap='gray')
        plt.title(f'True: {y_test[i]}, Pred: {y_pred[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.close()

    print(f"Neural Network Accuracy: {mlp.score(X_test_scaled, y_test) * 100:.2f}%")

def main():
    print("Machine Learning Digit Recognition")
    print("1. Logistic Regression on MNIST (28x28)")
    print("2. Neural Network on Digits Dataset (8x8)")
    choice = input("Enter 1 or 2 to select a model: ")

    if choice == '1':
        logistic_regression_mnist()
    elif choice == '2':
        neural_network_digits()
    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()
