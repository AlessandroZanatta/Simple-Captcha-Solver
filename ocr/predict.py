from keras.models import load_model
import numpy as np


# load trained model
model = load_model("ocr/model.model", compile=True)


def predict(images):
    """
        Expects a list of BS (batch size) images.
        Ouputs the predictions based on the loaded model
    """
    global model

    assert len(images) == 6 # change with BS
    images = np.expand_dims(images, axis=-1)

    prediction = model.predict(images)
    res = ""
    for i in range(len(images)):
        res += chr(ord("A") + np.where(prediction[i] == np.amax(prediction[i]))[0][0])

    return res
