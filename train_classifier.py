import os, shutil, random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import tensorflow_addons as tfa


# Count images for train valid test split
def get_sets_amount(valid_x, test_x, path_to_folder):
    count_images = 0

    folders = [x for x in os.listdir(path_to_folder) if not x.startswith(".")]
    for folder in folders:
        path = os.path.join(path_to_folder, folder)
        for image in os.listdir(path):
            image_path = os.path.join(path, image)
            if os.path.isfile(image_path) and not image.startswith("."):
                count_images += 1

    valid_amount = int(count_images * valid_x)
    test_amount = int(count_images * test_x)
    train_amount = count_images - valid_amount - test_amount

    return train_amount, valid_amount, test_amount


# Split images by folders
def create_sets_folders(path_to_folder, valid_part, test_part, classes):
    train_amount, valid_amount, test_amount = get_sets_amount(valid_part, test_part, path_to_folder)
    print(f'Train images: {train_amount}\nValid images: {valid_amount}\nTest images: {test_amount}')

    os.chdir(path_to_folder)
    if os.path.isdir('train') is False:

        os.mkdir('valid')
        os.mkdir('test')

        for name in classes:
            shutil.copytree(f'{name}', f'train/{name}')
            os.mkdir(f'valid/{name}')
            os.mkdir(f'test/{name}')

            valid_samples = random.sample(os.listdir(f'train/{name}'), round(valid_amount / len(classes)))
            for j in valid_samples:
                shutil.move(f'train/{name}/{j}', f'valid/{name}')

            test_samples = random.sample(os.listdir(f'train/{name}'), round(test_amount / len(classes)))
            for k in test_samples:
                shutil.move(f'train/{name}/{k}', f'test/{name}')

        print('Created train, valid and test directories')


# Load data from folders to iamge generator with batches
def load_data(path_to_folder, valid_part, test_part, classes, img_size, batch_size):
    create_sets_folders(path_to_folder, valid_part, test_part, classes)

    train_path = os.path.join(path_to_folder, 'train')
    valid_path = os.path.join(path_to_folder, 'valid')
    test_path = os.path.join(path_to_folder, 'test')

    preprocessing_function = keras.applications.efficientnet.preprocess_input

    train_batches = ImageDataGenerator(preprocessing_function=preprocessing_function).flow_from_directory(
        directory=train_path, target_size=(img_size, img_size),
        classes=classes, batch_size=batch_size
        )
    valid_batches = ImageDataGenerator(preprocessing_function=preprocessing_function).flow_from_directory(
        directory=valid_path, target_size=(img_size, img_size),
        classes=classes, batch_size=batch_size, shuffle=False
        )
    test_batches = ImageDataGenerator(preprocessing_function=preprocessing_function).flow_from_directory(
        directory=test_path, target_size=(img_size, img_size),
        classes=classes, batch_size=batch_size, shuffle=False
        )

    return train_batches, valid_batches, test_batches


# Load model and change architecture
def build_model(num_classes, layers_to_train, img_size):
    inputs = layers.Input(shape=(img_size, img_size, 3))
    model = keras.applications.EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")

    # Freese main layers (use N layers for training)
    for layer in model.layers[-layers_to_train:]:
        if not isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer='Adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', tfa.metrics.F1Score(num_classes)]
    )

    return model


def train(model, train_batches, valid_batches, epochs):
    early_stopping = keras.callbacks.EarlyStopping(
        patience=7,
        min_delta=0.001,
        restore_best_weights=True,
    )

    model.fit(
        x=train_batches,
        validation_data=valid_batches,
        epochs=epochs,
        callbacks=[early_stopping]
    )
    return model


def eval_model(model, test_batches):
    scores = model.evaluate(test_batches)
    print(f'Accuracy: {round((scores[1] * 100), 2)}%')


def main():
    path_to_folder = '/Users/argosaakyan/Data/dis_arm/classification/crops'
    path_to_save = 'classification/classificator_model'
    classes = ['small_gun', 'big_gun', 'phone', 'umbrella', 'empty']

    epochs = 20
    valid_part = 0.15
    test_part = 0.05
    layers_to_train = 23
    img_size = 224
    batch_size = 30

    train_batches, valid_batches, test_batches = load_data(path_to_folder, valid_part, test_part,
                                                           classes, img_size, batch_size)
    model = build_model(len(classes), layers_to_train, img_size)
    model = train(model, train_batches, valid_batches, epochs)
    eval_model(model, test_batches)

    model.save(path_to_save)


if __name__ == '__main__':
    main()
