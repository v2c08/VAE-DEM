def predict(latents):
	from keras.models import load_model, model_from_json
	import numpy as np
	import os
	from sklearn.preprocessing import MultiLabelBinarizer
	from itertools import product

	shapes = ['Cube', 'Cone', 'Cylinder', 'Icosphere', 'Torus']
	colours = ['r', 'g', 'b', 'w']

	labels = []
	for c, r in enumerate(product(colours, shapes)):
		labels.append(r)
	mlb = MultiLabelBinarizer()
	mlb.fit(labels)

	print(latents)
	print(os.getcwd())
	latents = np.reshape(latents, (1,len(latents)))

	json_file = open('../weights/classifier.json', 'r')

	loaded_model_json = json_file.read()

	json_file.close()

	classifier = model_from_json(loaded_model_json)

	classifier.load_weights('../weights/classifier.hdf5')

	out = classifier.predict(latents)
	print(out)
	print(out[0])
	out = mlb.inverse_transform(np.array(out))

	return out
