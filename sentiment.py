# Import TensorFlow library for machine learning tasks
import tensorflow as tf  # TensorFlow is an open-source software library for data analytics and machine learning developed by Google[1]. It provides tools to build and train neural networks, among other capabilities.

# Import Seaborn library for visualization of statistical data
import seaborn as plt  # Seaborn is a Python data visualization library built on top of Matplotlib. It offers high-level functions for creating attractive and informative statistical graphics[3][6].

# Import regular expression module for pattern matching in strings
import re  # Regular expressions (regex) are sequences of characters used to define search patterns within text. The `re` module in Python supports regex operations[2].

# Import shutil module for file system manipulation
import shutil  # The `shutil` module in Python provides convenience functions for copying, moving, and removing files and directories[4].

# Import string module for working with character strings
import string  # The `string` module in Python contains constants representing all ASCII printing characters, digits, punctuation marks, etc., which can be useful when dealing with strings or generating random strings[5].
import os

from tensorflow.keras import layers
from tensorflow.keras import losses

print("tensorflow version = ", tf.__version__)


# downloading the IMDB dataset

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset = tf.keras.utils.get_file(
    "aclImdb_v1", url, untar=True, cache_dir=".", cache_subdir="."
)


dataset_dir = os.path.join(os.path.dirname(dataset), "aclImdb")

os.listdir(dataset_dir)

train_dir = os.path.join(dataset_dir, "train")
os.listdir(train_dir)

sample_file = os.path.join(train_dir, "pos/1181_9.txt")
with open(sample_file) as f:
    print(f.read())

remove_dir = os.path.join(train_dir, "unsup")
""" us it when running for the first time"""
shutil.rmtree(remove_dir)

# Set the batch size for the training dataset
batch_size = 32

# Set the random seed for reproducibility
seed = 42

# Load the training dataset from the 'aclImdb/train' directory using TensorFlow's `text_dataset_from_directory()` function
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    "aclImdb/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=seed,
)

for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(1):
        print("Review", text_batch.numpy()[i])
        print("Label", label_batch.numpy()[i])

print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])

raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    "aclImdb/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=seed,
)

print(raw_train_ds.class_names[0])


def custom_standardization(input_data):
    # Convert input text to lowercase
    lowercase = tf.strings.lower(input_data)

    # Remove HTML line breaks '<br />' and replace them with spaces
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")

    # Remove punctuation from the text using regular expression
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape(string.punctuation), ""
    )


"""Next, you will create a TextVectorization layer. You will use this layer to standardize, tokenize, and vectorize our data. You set the output_mode to int to create unique integer indices for each token.

Note that you're using the default split function, and the custom standardization function you defined above. You'll also define some constants for the model, like an explicit maximum sequence_length, which will cause the layer to pad or truncate sequences to exactly sequence_length values."""


# Set the maximum number of features to use in the vectorization layer
max_features = 10000

# Set the maximum sequence length for the vectorization layer
sequence_length = 250

# Define a TextVectorization layer with the specified parameters
vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,  # Use the custom standardization function defined earlier
    max_tokens=max_features,  # Set the maximum number of features to use
    output_mode="int",  # Output integer-encoded sequences
    output_sequence_length=sequence_length,
)  # Set the maximum sequence length


# Make a text-only dataset (without labels), then call adapt
# Process the raw text data from the training dataset using a lambda function
train_text = raw_train_ds.map(lambda x, y: x)

# Adapt the TextVectorization layer to the training text data
vectorize_layer.adapt(train_text)


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


# retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Label", raw_train_ds.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))

print("366 ----> ", vectorize_layer.get_vocabulary()[366])

raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    "aclImdb/test", batch_size=batch_size
)

# Map the raw training dataset using the vectorize_text function
train_ds = raw_train_ds.map(vectorize_text)

# Map the raw validation dataset using the vectorize_text function
val_ds = raw_val_ds.map(vectorize_text)

# Map the raw testing dataset using the vectorize_text function
test_ds = raw_test_ds.map(vectorize_text)

"""These are two important methods you should use when loading data to make sure that I/O does not become blocking.

.cache() keeps data in memory after it's loaded off disk. This will ensure the dataset does not become a bottleneck while training your model. If your dataset is too large to fit into memory, you can also use this method to create a performant on-disk cache, which is more efficient to read than many small files.

.prefetch() overlaps data preprocessing and model execution while training."""

# Define AUTOTUNE to dynamically adjust the number of parallel calls based on available CPU resources
AUTOTUNE = tf.data.AUTOTUNE

# Cache and prefetch the training dataset for improved performance
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Cache and prefetch the validation dataset for improved performance
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Cache and prefetch the testing dataset for improved performance
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Define the embedding dimension for the Embedding layer
embedding_dim = 16

# Define the machine learning model using a Sequential model
model = tf.keras.Sequential(
    [
        # Add an Embedding layer with the specified maximum number of features and embedding dimension
        layers.Embedding(max_features, embedding_dim),
        # Add a Dropout layer to prevent overfitting
        layers.Dropout(0.2),
        # Add a GlobalAveragePooling1D layer to reduce the dimensionality of the output
        layers.GlobalAveragePooling1D(),
        # Add another Dropout layer to prevent overfitting
        layers.Dropout(0.2),
        # Add a Dense layer with a single output unit for binary classification
        layers.Dense(1),
    ]
)


model.summary()

# Compile the model with the following configurations
model.compile(
    # Specify the loss function as Binary Cross Entropy
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    # Specify the optimizer as Adam
    optimizer="adam",
    # Specify the metric as Binary Accuracy
    metrics=[tf.metrics.BinaryAccuracy(threshold=0.0)],
)

epochs = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)


# Evaluate the model on the test dataset and retrieve the loss and accuracy
loss, accuracy = model.evaluate(test_ds)

# Print the loss and accuracy as percentages
print("loss:", str(loss * 100) + "%")
print("accuracy:", str(accuracy * 100) + "%")

export_model = tf.keras.Sequential(
    [vectorize_layer, model, layers.Activation("sigmoid")]
)

export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False),
    optimizer="adam",
    metrics=["accuracy"],
)

# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = export_model.evaluate(raw_test_ds)
print(accuracy)

# Saving the model
project_file=os.getcwd()
sentiment_model_fdname='sentiment_model'
sentiment_model=os.path.join(project_file,sentiment_model_fdname)
tf.saved_model.save(export_model,sentiment_model)