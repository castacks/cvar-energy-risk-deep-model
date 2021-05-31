import os
from data_utils import *
from deep_energy_model import *
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

class ValidationCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, val_data):
        super(ValidationCallback, self).__init__()
        # data to use as validation set, this should be dictionary
        # of the form {flight (int): tf.data.Dataset}
        self.val_data = val_data
        self.loss_monitor = []
    
    # this function is for calculating and recording (train loss, validation loss)
    def on_epoch_end(self, epoch, logs=None):
        training_loss = logs.get('loss')
        training_rmse = logs.get('root_mean_squared_error')
        training_mape = logs.get('mean_absolute_percentage_error')
        
        validation_loss = []
        for flight in self.val_data:
            self.model.reset_states()
            loss = self.model.evaluate(self.val_data[flight], verbose=0)
            if loss[2] <= 10000: # very high value; for instability
                validation_loss.append(loss)

        validation_loss = np.average(validation_loss, axis=0)
        self.loss_monitor.append([epoch, training_loss, training_rmse, training_mape, validation_loss[0], validation_loss[1], validation_loss[2]])
        
    def _get_values(self):
        return self.loss_monitor
    
def write_csv(target_dir, loss_data, epoch_data):
    result = np.concatenate((loss_data, epoch_data), axis=1)
    
    # check if the path exists
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    print(target_dir+'/csv_result.csv')
    np.savetxt(fname=target_dir+'/csv_result.csv', X=result, delimiter=',', fmt=['%d']+['%.6f']*6+['%d']*3, header='Epoch,Training Loss,Training RMSE,Training MAPE,Validation Loss,Validation RMSE,Validation MAPE,Epoch Index,Training Step,Flight', comments='')
    
# if the file is execeuted as the main program
if __name__ == '__main__':
    
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", default='./', help="Working directory for data.")
    parser.add_argument("-o", "--output", default='./', help="Output directory for training.")
    parser.add_argument("-D", "--data", action="store_true", help="Whether to download the data or not.")
    parser.add_argument("--lookback", default=20, type=int, help="Size of lookback window")
    parser.add_argument("--batch_size", default=32, type=int, help="Size of batch passed to `tf`.")
    parser.add_argument("--nb_filters", default=32, type=int, help="(For TCN) The number of filters to use in the convolutional layers. Would be similar to units for LSTM. Can be a list.")
    parser.add_argument("--kernel_size", default=3, type=int, help="(For TCN) The size of the kernel to use in each convolutional layer.")
    parser.add_argument("--nb_stacks", default=1, type=int, help="(For TCN) The number of stacks of residual blocks to use.")
    parser.add_argument("--n_layers", default=4, type=int, help="Number of layers in the network.")
    parser.add_argument("--total_epochs", default=10, type=int, help="Total number passes over the entire training data set.")
    parser.add_argument("--optimizer", default='Adam', help="Optimizer used by the neural network.")
    parser.add_argument("--dropout", default=0., type=float, help="Dropout used by the neural network.")
    parser.add_argument("--units", default=32, type=int, help="(For LSTM) The number of units in each layer.")
    parser.add_argument("--net_type", default='tcn', choices=['lstm', 'tcn'], help="The type of net to train. Either `lstm ` or `gru`.")
    parser.add_argument("--stateful", action="store_true", help="Whether to create a stateful neural net or not.")
    args = parser.parse_args()
    
    # validate directory
    directory = os.path.join(args.directory, '')
    if not os.path.exists(directory):
        raise ValueError('Invalid data path')
    
    # download dataset if needed
    if args.data:
        get_dataset(directory)
    
    # load the data
    all_data = load_dataset(directory+'data/')
    
    # load the net
    config = {k:v for k, v in vars(args).items() if k not in ['directory', 'output', 'data']}
    inputs = load_inputs(**config)
    
    # prepare tensors for training
    data, data_min, data_max, test_range, train_range, val_range = process_data(all_data, **inputs)
    train_tensors = create_tensors(data, train_range, drop_remainer=False, **inputs)
    val_tensors = create_tensors(data, val_range, drop_remainer=False, **inputs)
    
    # prepare name of storage
    if inputs['net_type'] == 'lstm':
        save_str = '{net_type}_{lookback}_{dropout}_{units}_{n_hidden}_{optimizer}'.format(**inputs)
    elif inputs['net_type'] == 'tcn':
        save_str = '{net_type}_{lookback}_{nb_filters}_{kernel_size}_{nb_stacks}_{n_layers}'.format(**inputs)

    # instantiate stuff for this run
    model = create_advanced_net((inputs['lookback'], 5), **inputs)
    model_dir = output_dir+'Results/saved_models/'+save_str
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model.save(model_dir)

    # for tensorboard
    log_dir = output_dir+'Results/logs/'+save_str+'/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    # for model checkpoint
    checkpoint_path = output_dir+'Results/ckpts/'+save_str+'/cp-{epoch:04d}.ckpt'

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        period=50)

    model.save_weights(checkpoint_path.format(epoch=0))

    # for validation calculation
    val_callback = ValidationCallback(val_data=val_tensors)

    epoch_list = []

    # training loop
    for epoch_idx in range(inputs['total_epochs']):

        np.random.seed(42)
        np.random.shuffle(train_range)

        if epoch_idx == 0:
            initial_epoch = 0

        for idx, flight in enumerate(train_range):
            history = model.fit(train_tensors[flight], callbacks=[val_callback, cp_callback, tensorboard_callback], batch_size=inputs['batch_size'], epochs=1+initial_epoch, verbose=0, initial_epoch=initial_epoch)
            initial_epoch += 1
            epoch_list.append([epoch_idx, idx, flight])
            model.reset_states()

    # dir for storing csv results
    csv_dir = output_dir+'Results/CsvResults/'+save_str
    epoch_data = np.array(epoch_list)
    loss_data = np.array(val_callback._get_values())
    write_csv(csv_dir, loss_data, epoch_data)