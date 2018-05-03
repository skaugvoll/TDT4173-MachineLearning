from convolutional_network_tf import init_model, change_num_input, train_conv_net, test_conv_net, prediction_conv_net
from data_generator import DataGenerator
import numpy as np
import tensorflow as tf


def ocr(trainingset="../chars74k-lite", testingset="../detection-images"):
    #########
    #
    # INITIALIZATION
    #
    #########
    window_character_threshold = 0.95
    patches_containing_charater = []
    characters = set()


    # TRAINING CASES
    dg = DataGenerator(dataset=trainingset, normalized=True)
    dg.shuffle_data()

    # extract the training  data from our data-generator
    training_cases, training_labels = dg.get_training_partition(percentage=1)

    # convert training  data from float64 to float32
    training_cases = training_cases.astype(np.float32)

    # print(training_cases[0].shape) # (400,)

    # TESTING CASES
    dg = DataGenerator(dataset=testingset, normalized=True)

    # extract the testing  data from our data-generator
    testing_cases,_ = dg.get_data()
    testing_labels = training_labels


    # print(np.array(testing_cases[0]).shape) # (40000,)


    cases_reshaped = []
    for idx, case in enumerate(testing_cases):
        h,w = dg.get_dimensions(idx)
        case = np.array(case)
        case = case.reshape(h,w)
        cases_reshaped.append(case)


    # print(cases_reshaped[0].shape) # (200,200) # (600,300)


    # windows = window_stack(cases_reshaped[0], 1, 20)

    # print(windows.shape) # (181,4000)

    # print(windows[0].shape) # (4000,)

    #########
    #
    # OCR
    #
    #########

    # create the classifier to run over the windows
    # Create the Conv model
    print("Creating model")
    window_width = 20
    window_height = 20
    image_width = 200
    image_height = 200

    model = init_model(window_height, window_width)

    # Train the model
    print("Training model")
    model = train_conv_net(model, training_cases, training_labels)

    print("Creating windows")
    columns = image_width - window_width
    rows = image_height - window_height
    windows = []
    for i in range(columns):
        for j in range(rows):
            index = i + j * image_width
            windows.append(get_window(testing_cases[0], index, window_width, window_height, image_width).reshape(1, 400))

    windows = np.array(windows)
    print("Predicting windows")
    template = ('Prediction is "{}" ({:.1f}%)')
    predictions = prediction_conv_net(model, windows)
    for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        if probability > window_character_threshold:
            # print(template.format(dg.int_to_char(class_id), 100 * probability))
            characters.add(dg.int_to_char(class_id))

    print("Characters found with threshold " + str(window_character_threshold) + ":" + str(characters))

    # print(len(get_window(testing_cases[0], 0, window_width, window_height, 200)))

    # for label_idx, case in enumerate(cases_reshaped[0]):
    #     windows = window_stack(case, 1, 2)
    #     windows = windows.astype(np.float32)
    #     print(windows.shape, windows[0].shape)
    #     for window in windows:
    #         window = window.reshape(1,400)
    #
    #         # print(window, window.shape, [testing_labels[label_idx]])
    #         model, estimator_generator, prob = prediction_conv_net(model, window, np.array([testing_labels[label_idx]]))


def get_window(array, index, window_width, window_height, img_width):
    window = []
    current_index = index
    for i in range(window_height):
        window.extend(array[current_index:current_index + window_width])
        current_index += img_width

    return np.array(window, np.float32)


def window_stack(a, stepsize=1, width=3):
    n = a.shape[0]
    return np.hstack( a[i:1+n+i-width : stepsize] for i in range(0, width) )



def notMain():
    # get the data
    dg = DataGenerator(dataset="../chars74k-lite", normalized=True)
    dg.shuffle_data()

    # extract the training and testing data from our data-generator
    training_cases, training_labels = dg.get_training_partition(percentage=.8)
    testing_cases, testing_labels = dg.get_testing_partition(percentage=.2)

    # convert training and testing data from float64 to float32 because the conv - needs float32
    training_cases = training_cases.astype(np.float32)
    testing_cases = testing_cases.astype(np.float32)


    # Create the Conv model
    print("Creating model")
    img_height = 20
    img_width = 20
    model = init_model(img_height, img_width)


    # Train the model
    print("Training model")
    train_conv_net(model, training_cases, training_labels)

    # Test the model
    print("Testing model")
    _, accuracy = test_conv_net(model, testing_cases, testing_labels)

    print("Accuracy: {}".format(accuracy))

    # Prediction by the model
    print("now prediction using the model")
    model, estimator_generator, prob = prediction_conv_net(model, testing_cases, testing_labels)
    print("Predictions probabilities:\n{}\n".format(prob))




if __name__ == "__main__":
    # notMain()
    ocr()