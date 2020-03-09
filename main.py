import pprint
import logging
from tools.loader import get_loaders
from tools.model_factory import Model
from tools.misc import load_cf, set_logger

def main(yaml_filepath):

    ### Logging lines ###
    cf = load_cf(yaml_filepath)

    # Print the configuration - just to make sure that you loaded what you
    # wanted to load
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(cf)
    #Set the logger to write the logs of the training in train.log
    set_logger(cf)
    logging.info(' --------- Init experiment: ' + cf['exp_name'] + ' ---------')


    ### Create the data generators ###
    logging.info('\n > Creating data generators...')
    train_gen, val_gen = get_loaders(cf)


    ### Build model ###
    logging.info('\n > Building model...')
    model = Model(cf)


    if cf['train']:
        print()
        ### Training the model ###
        model.train(train_gen, val_gen)      


    if cf['test']:
        print()
        ### Compute validation metrics ###
        model.test(val_gen)

    # Finish
    logging.info(' --------- Finish experiment: ' + cf['exp_name'] + ' ---------')


# Entry point of the script
if __name__ == "__main__":
    main('config.yaml')