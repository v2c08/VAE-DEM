def predict(image):
    from ../model_util import SampleLayer
    from keras.models import load_model, model_from_json
    import numpy as np
    image = np.asarray(image)
    json_file = open('weights/encoder.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    encoder = model_from_json(loaded_model_json, custom_objects={'SampleLayer': SampleLayer})
    encoder.load_weights("weights/encoder.h5")
    out = encoder.predict(image)
    return out[0]
