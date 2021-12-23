import sys
import trainer

params = {
    'model_name': '', # model_pad_fill.h5, model.h5
    'num_start' : 11, # 11: excluding 0-9 digits from dataset
    'num_end' : 62, # 62
    'num_data' : 55, # number of dataset for each character
    'num_train' : 40, # number of training set
}

def modeler(model_name):
    params['model_name'] = model_name
    training = trainer.trainer(params)
    training.train_dataset()

# run python command line (e.g. python3.7 modeler.py model_pad_fill.h5)
modeler(sys.argv[1])
