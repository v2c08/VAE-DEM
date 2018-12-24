def predict(latents):
    from models import SampleLayer
    from keras.models import load_model, model_from_json
    import numpy as np

    print(latents)
    latents = np.reshape(latents, (1,len(latents)))
    json_file = open('weights/decoder.json', 'r')

    loaded_model_json = json_file.read()

    json_file.close()

    decoder = model_from_json(loaded_model_json)

    decoder.load_weights("weights/decoder.h5")

    out = decoder.predict(latents)

    return out[0]
