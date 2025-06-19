import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os

# Constants
IMG_HEIGHT = 48
IMG_WIDTH = 48
IMG_CHANNELS = 1 # Grayscale
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
BATCH_SIZE = 64
EPOCHS = 50 # Increased epochs, rely on EarlyStopping
NUM_CLASSES = 6
PATIENCE_EARLY_STOPPING = 10
PATIENCE_REDUCE_LR = 5
MODEL_SAVE_PATH = 'emotion_model_cnn_scratch.keras' # Use .keras format

# Define paths - Adjust if necessary based on workspace structure
base_dir = '.' # Assuming script is run from the workspace root
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Check if directories exist
if not os.path.isdir(train_dir):
    raise ValueError(f"Training directory not found: {train_dir}")
if not os.path.isdir(test_dir):
    raise ValueError(f"Testing directory not found: {test_dir}")


# --- 1. Data Loading and Preprocessing ---
print("--- Loading Data ---")
try:
    # Training dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='categorical', # Use categorical for softmax output
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        interpolation='nearest',
        batch_size=BATCH_SIZE,
        color_mode='grayscale', # Convert images to grayscale
        shuffle=True
    )

    # Validation dataset (using the 'test' directory as validation during training)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='categorical',
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        interpolation='nearest',
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        shuffle=False # No need to shuffle validation data
    )
except Exception as e:
    print(f"Error loading data: {e}")
    print("Please ensure the 'train' and 'test' directories exist and contain subdirectories for each emotion.")
    exit()

# Get class names (emotions)
class_names = train_ds.class_names
print(f"Found classes: {class_names}")
if len(class_names) != NUM_CLASSES:
    print(f"Warning: Expected {NUM_CLASSES} classes, but found {len(class_names)} in {train_dir}")
    print("Please ensure the training directory has subfolders for all expected emotions.")
    # Potentially adjust NUM_CLASSES or raise an error depending on desired behavior
    # NUM_CLASSES = len(class_names) # Adjust if needed
    # exit()


# --- 2. Data Augmentation and Normalization ---
print("--- Configuring Data Preprocessing ---")
# Normalization layer (rescale pixel values from 0-255 to 0-1)
rescale_layer = layers.Rescaling(1./255)

# Data augmentation layers (applied only during training)
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        # layers.RandomContrast(0.1), # Optional: Add contrast/brightness if needed
    ],
    name="data_augmentation",
)

# Apply normalization and augmentation
# Note: Augmentation is applied *after* rescaling
train_ds = train_ds.map(lambda x, y: (rescale_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

# Apply only normalization to validation set
val_ds = val_ds.map(lambda x, y: (rescale_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

# Configure datasets for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# --- 3. Build the CNN Model (From Scratch) ---
print("--- Building Model ---")
model = keras.Sequential([
    # Input shape defined in the first layer
    layers.Input(shape=IMG_SHAPE),

    # --- Conv Block 1 ---
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    # --- Conv Block 2 ---
    layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    # --- Conv Block 3 ---
    layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    # --- Fully Connected Layers ---
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax') # Output layer
], name="CNN_Emotion_Scratch")

model.summary()

# --- 4. Compile the Model ---
print("--- Compiling Model ---")
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # Start with a common LR
loss_function = tf.keras.losses.CategoricalCrossentropy()

model.compile(optimizer=optimizer,
              loss=loss_function,
              metrics=['accuracy'])

# --- 5. Define Callbacks ---
print("--- Setting Up Callbacks ---")
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH,
                             monitor='val_accuracy',
                             save_best_only=True,
                             mode='max',
                             verbose=1)

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=PATIENCE_EARLY_STOPPING,
                               restore_best_weights=True, # Restore model weights from the epoch with the best value of the monitored quantity.
                               verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2, # Reduce LR by factor of 5
                              patience=PATIENCE_REDUCE_LR,
                              min_lr=0.00001, # Don't reduce LR below this
                              verbose=1)

callbacks_list = [checkpoint, early_stopping, reduce_lr]

# --- 6. Train the Model ---
print("\n--- Starting Training ---")
try:
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks_list
    )
    print("--- Training Finished ---")
except Exception as e:
    print(f"An error occurred during training: {e}")
    exit()

# --- 7. Evaluate the Best Model (Loaded by EarlyStopping's restore_best_weights=True) ---
print("\n--- Evaluating Model on Test Set ---")
try:
    test_loss, test_acc = model.evaluate(val_ds, verbose=1) # Evaluate on the val_ds (which is our test set)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.4f}')
except Exception as e:
    print(f"An error occurred during evaluation: {e}")

# --- 8. Final Save Confirmation ---
print(f"\nBest model based on validation accuracy saved to: {MODEL_SAVE_PATH}")

# --- 9. Optional: Plot training history ---
# import matplotlib.pyplot as plt
# def plot_history(history):
#     if not history: return # Handle case where training might have failed early
#     acc = history.history.get('accuracy')
#     val_acc = history.history.get('val_accuracy')
#     loss = history.history.get('loss')
#     val_loss = history.history.get('val_loss')

#     if not all([acc, val_acc, loss, val_loss]):
#         print("Could not plot history: Missing metrics.")
#         return

#     epochs_range = range(len(acc)) # Use actual length in case of early stopping

#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs_range, acc, label='Training Accuracy')
#     plt.plot(epochs_range, val_acc, label='Validation Accuracy')
#     plt.legend(loc='lower right')
#     plt.title('Training and Validation Accuracy')

#     plt.subplot(1, 2, 2)
#     plt.plot(epochs_range, loss, label='Training Loss')
#     plt.plot(epochs_range, val_loss, label='Validation Loss')
#     plt.legend(loc='upper right')
#     plt.title('Training and Validation Loss')
#     plt.show()

# if 'history' in locals():
#    plot_history(history)

print("\nScript finished.") 