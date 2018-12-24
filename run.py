import os
import argparse
from models import Models
import yaml

WEIGHTS_DIR  = os.path.join(os.getcwd(), 'weights')
if not os.path.exists(WEIGHTS_DIR): os.makedirs(WEIGHTS_DIR)
DATA_DIR     = os.path.join(os.getcwd(),'data')
if not os.path.exists(DATA_DIR):    os.makedirs(DATA_DIR)
RESULTS_DIR  = os.path.join(os.getcwd(),'results')
if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)


if __name__  == '__main__' :

	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', default='train', choices=['train','test'])
	parser.add_argument('--gendata', default=False)

	args = parser.parse_args()

	P = yaml.load(open('parameters.txt'))

	if args.gendata:
		import world

	models = Models(P)

	if args.mode == 'train':
		models.build_models()
		models.train_bvae()
		models.train_classifier()
	if args.mode == 'test':
		models.eval_bvae()
		models.eval_classifier()
