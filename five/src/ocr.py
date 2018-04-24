from convolutional_network_tf import init_model, train_conv_net, test_conv_net
from data_generator import DataGenerator
import numpy as np



def main():
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
    model = init_model()


    # Train the model
    print("Training model")
    train_conv_net(model, training_cases, training_labels)

    # Test the model
    print("Testing model")
    _, accuracy = test_conv_net(model, testing_cases, testing_labels)


    print("Accuracy: {}".format(accuracy))



if __name__ == "__main__":
    main()