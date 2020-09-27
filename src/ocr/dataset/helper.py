import numpy as np


def load_dataset(path):
    # initialize the list of data and labels
    data = []
    labels = []

    # retrieve data from CSV
    for row in open(path):
        # parse the label and image from the row
        row = row.split(",")
        label = ord(row[0]) - ord("A") # scale labels to be numbers in 0-25 (26 letters)
        image = np.array([int(x) for x in row[1].split(".")], dtype="uint8")

        # Reshape flattened images
        image = image.reshape((32, 32))
        # update the list of data and labels
        data.append(image)
        labels.append(label)

    return (data, labels)

