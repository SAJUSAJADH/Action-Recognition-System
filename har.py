import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset paths
train_path = "dataset/Human_Action_Recognition/train"
test_path = "dataset/Human_Action_Recognition/test"
train_csv_path = "dataset/Human_Action_Recognition/Training_set.csv"
test_csv_path = "dataset/Human_Action_Recognition/Testing_set.csv"

# Load CSV files
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

# Store the original categories
categories = train_df['label'].astype('category').cat.categories

# Distribution of classes in training set
fig = px.histogram(train_df, x='label', title='Distribution of Classes in Training Set')
fig.show()

# Images from each class
fig, axes = plt.subplots(3, 5, figsize=(20, 10))
axes = axes.flatten()
for idx, class_name in enumerate(train_df['label'].unique()):
    class_images = train_df[train_df['label'] == class_name]['filename'].values
    img = plt.imread(os.path.join(train_path, class_images[0]))
    axes[idx].imshow(img)
    axes[idx].set_title(class_name)
    axes[idx].axis('off')
plt.tight_layout()
plt.show()

# Data Preprocessing
train_df['label'] = train_df['label'].astype('category')
train_df['label'] = train_df['label'].cat.codes
train_df['filepath'] = train_df['filename'].apply(lambda x: os.path.join(train_path, x))

# Split training and validation set
train_set, val_set = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=42)

def load_image(filepath, label):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [128, 128])
    image = image / 255.0
    return image, label

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_set['filepath'].values, train_set['label'].values))
train_dataset = train_dataset.map(load_image).batch(32).shuffle(buffer_size=len(train_set))

val_dataset = tf.data.Dataset.from_tensor_slices((val_set['filepath'].values, val_set['label'].values))
val_dataset = val_dataset.map(load_image).batch(32)

# Model Building
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model Training
history = model.fit(train_dataset, validation_data=val_dataset, epochs=30)

# Plot Training and Validation Accuracy
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(history.history['accuracy'])), y=history.history['accuracy'], mode='lines', name='Training Accuracy'))
fig.add_trace(go.Scatter(x=np.arange(len(history.history['val_accuracy'])), y=history.history['val_accuracy'], mode='lines', name='Validation Accuracy'))
fig.update_layout(title='Model Accuracy', xaxis_title='Epoch', yaxis_title='Accuracy')
fig.show()

# Plot Training and Validation Loss
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(history.history['loss'])), y=history.history['loss'], mode='lines', name='Training Loss'))
fig.add_trace(go.Scatter(x=np.arange(len(history.history['val_loss'])), y=history.history['val_loss'], mode='lines', name='Validation Loss'))
fig.update_layout(title='Model Loss', xaxis_title='Epoch', yaxis_title='Loss')
fig.show()

# Predictions on validation set
val_predictions = np.argmax(model.predict(val_dataset), axis=-1)
val_labels = np.concatenate([y.numpy() for x, y in val_dataset], axis=0)

# Classification Report
print(classification_report(val_labels, val_predictions, target_names=categories))

# Confusion Matrix
conf_matrix = confusion_matrix(val_labels, val_predictions)
fig = px.imshow(conf_matrix, text_auto=True, title='Confusion Matrix')
fig.show()

# Save model
model.save('human_action_recognition_model.h5')

# Prepare test data
test_df['filepath'] = test_df['filename'].apply(lambda x: os.path.join(test_path, x))

test_dataset = tf.data.Dataset.from_tensor_slices(test_df['filepath'].values)
test_dataset = test_dataset.map(lambda x: (tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(x), channels=3), [128, 128]) / 255.0)).batch(32)

# Predict on test set
test_predictions = np.argmax(model.predict(test_dataset), axis=-1)
test_df['label'] = test_predictions

# Map label codes back to class names
label_map = {i: label for i, label in enumerate(categories)}
test_df['label'] = test_df['label'].map(label_map)

# Prepare submission
submission = test_df[['filename', 'label']]
submission.to_csv('submission.csv', index=False)

# Show 10 images with actual and predicted labels from validation set
fig, axes = plt.subplots(2, 5, figsize=(20, 10))
axes = axes.flatten()
sampled_val_set = val_set.sample(10, random_state=42)
for i, (index, row) in enumerate(sampled_val_set.iterrows()):
    img = plt.imread(row['filepath'])
    actual_label = categories[row['label']]
    predicted_label = categories[val_predictions[val_set.index.get_loc(index)]]
    axes[i].imshow(img)
    axes[i].set_title(f"Actual: {actual_label}\nPredicted: {predicted_label}")
    axes[i].axis('off')
plt.tight_layout()
plt.show()
