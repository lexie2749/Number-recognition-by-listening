import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

# Define the path to your images folder
image_folder = 'spectrograms'  # Update with the correct path
image_size = (128, 128)  # Resize images to 128x128

# Create lists to hold images and labels
images = []
labels = []

# Load images and labels based on filename pattern
for filename in os.listdir(image_folder):
    if filename.endswith('.png'):
        # Extract label from filename (first number before the dash)
        label = int(filename.split('-')[0])
        
        # Only include labels between 0 and 10
        if 0 <= label <= 10:
            labels.append(label)
            
            # Load image and convert to RGB
            img_path = os.path.join(image_folder, filename)
            img = Image.open(img_path).resize(image_size).convert('RGB')  # Convert to RGB
            img_array = np.array(img) / 255.0  # Normalize pixel values
            images.append(img_array)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split the data into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images,
                                                                        labels,
                                                                        test_size=0.1, random_state=42)

# Define the CNN model with correct input shape
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),  # Adjusted input shape for 128x128 images
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(11, activation='softmax')  # Adjust output layer to match the number of classes (0-10 => 11 classes)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, batch_size=8, epochs=50,
                    validation_data=(test_images, test_labels))

# Increase plot size
plt.figure(figsize=(12, 8))  # Set figure size (width, height)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')

# Set labels and legend with Times New Roman and font size 24
plt.xlabel('Epoch', fontsize=24, fontname='Times New Roman')
plt.ylabel('Accuracy', fontsize=24, fontname='Times New Roman')

# Adjust axis limits
plt.xlim([0, 50])  # Adjusted to 50 epochs since the training used 50 epochs
plt.ylim([0.0, 1.0])

# Remove title
# Set font for tick marks
plt.xticks(fontsize=24, fontname='Times New Roman')
plt.yticks(fontsize=24, fontname='Times New Roman')

plt.legend(loc='lower right', fontsize=24, prop={'family': 'Times New Roman'})

# Save the plot
plt.savefig('result-specto-custom.png')
 
# Show the plot
plt.show()

# Import necessary libraries
import numpy as np

# Evaluate the model on the test set and output predicted labels
def evaluate_with_custom_accuracy_and_std(test_images, test_labels):
    """Evaluate the model and output predicted and true labels, along with custom accuracy calculation."""
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Create a list to store the custom accuracies for calculating standard deviation
    custom_accuracies = []
    
    # Iterate through the test set to compare predicted and true labels
    for i in range(len(test_labels)):
        true_label = test_labels[i]
        predicted_label = predicted_labels[i]
        
        # Compute the accuracy with the given formula
        if true_label + predicted_label != 0:  # Avoid division by zero
            custom_accuracy = (predicted_label - true_label) / (predicted_label + true_label)
        else:
            custom_accuracy = 0.0
        
        # Append custom accuracy to the list
        custom_accuracies.append(custom_accuracy)
        
        print(f"True label: {true_label}, Predicted label: {predicted_label}, Custom accuracy: {custom_accuracy}")
    
    # Convert list to a numpy array for standard deviation calculation
    custom_accuracies = np.array(custom_accuracies)
    
    # Calculate the standard deviation of custom accuracies
    std_dev = np.std(custom_accuracies)
    print(f"Standard deviation of custom accuracies: {std_dev}")

# Run the custom evaluation and standard deviation calculation
evaluate_with_custom_accuracy_and_std(test_images, test_labels)

# Evaluate the model on the test set using standard accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test accuracy: {test_acc}")
