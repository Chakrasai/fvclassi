import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
# Set the paths for your dataset
train_data_path = r'C:\Users\chakr\OneDrive\Desktop\dataset\train'           
validation_data_path = r'C:\Users\chakr\OneDrive\Desktop\dataset\validation' 
test_data_path = r'C:\Users\chakr\OneDrive\Desktop\dataset\test'            

# Print the paths to verify
print(f"Train Data Path: {train_data_path}")
print(f"Validation Data Path: {validation_data_path}")
print(f"Test Data Path: {test_data_path}")

# Check if directories exist
if not os.path.exists(train_data_path):
    print(f"Training data path does not exist: {train_data_path}")
if not os.path.exists(validation_data_path):
    print(f"Validation data path does not exist: {validation_data_path}")
if not os.path.exists(test_data_path):
    print(f"Test data path does not exist: {test_data_path}")

# Data preparation and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2]
    
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_data_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Get the number of classes
num_classes = len(train_generator.class_indices)

# Model building
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu',input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu',input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(256, (3, 3), activation='relu',input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Update the number of classes here
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with early stopping and model checkpoint
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
]

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=callbacks
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save the model for future use
model.save('fruit_vegetable_classifier.keras')

# Function to classify new images
def classify_image(img_path):
    model = load_model('fruit_vegetable_classifier.keras')
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    prediction = model.predict(img_array)
    classes = train_generator.class_indices
    predicted_class = list(classes.keys())[np.argmax(prediction)]
    return predicted_class

# Example usage of the classify_image function
input_image_path = r'C:\Users\chakr\OneDrive\Desktop\dataset\train\apple\Image_2.jpg'  
predicted_class = classify_image(input_image_path)
print(f"This image is likely a {predicted_class}")
