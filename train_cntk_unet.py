from __future__ import print_function

import numpy as np

import cntk as C
from cntk.device import try_set_default_device, gpu
from cntk.learners import learning_rate_schedule, UnitType


import simulation
import cntk_unet
import helper


try_set_default_device(gpu(0))

def slice_minibatch(data_x, data_y, i, minibatch_size):
    sx = data_x[i * minibatch_size:(i + 1) * minibatch_size]
    sy = data_y[i * minibatch_size:(i + 1) * minibatch_size]

    return sx, sy

def measure_loss(data_x, data_y, x, y, trainer, minibatch_size):
    errors = []
    for i in range(0, int(len(data_x) / minibatch_size)):
        data_sx, data_sy = slice_minibatch(data_x, data_y, i, minibatch_size)

        errors.append(trainer.test_minibatch({x: data_sx, y: data_sy}))

    return np.mean(errors)

def train(input_images, target_masks, use_existing=False):
    shape = input_images[0].shape
    data_size = input_images.shape[0]
    
    # Split data
    test_portion = int(data_size * 0.1)
    indices = np.random.permutation(data_size)
    test_indices = indices[:test_portion]
    training_indices = indices[test_portion:]

    validation_data = (input_images[test_indices], target_masks[test_indices])
    training_data = (input_images[training_indices], target_masks[training_indices])

    # Construct the model
    x = C.input_variable(shape)
    y = C.input_variable(target_masks[0].shape)

    print(y)
    z = cntk_unet.create_model(x, target_masks.shape[1])
    dice_coef = cntk_unet.dice_coefficient(z, y)

    checkpoint_file = "cntk-unet.dnn"
    if use_existing:
        z.load_model(checkpoint_file)

    # Prepare model and trainer
    lr = learning_rate_schedule(0.00001, UnitType.sample)
    momentum = C.learners.momentum_as_time_constant_schedule(0)
    trainer = C.Trainer(z, (-dice_coef, -dice_coef), C.learners.adam(z.parameters, lr=lr, momentum=momentum))

    # Get minibatches of training data and perform model training
    minibatch_size = 4
    num_epochs = 10
    num_mb_per_epoch = int(data_size / minibatch_size)
    
    # Record loss variation
    training_loss_list = []
    validation_loss_list = []

    print("Training_X : {}".format(training_data[0].shape))
    print("Training_Y : {}".format(training_data[1].shape))
    
    
    for e in range(0, num_epochs):
        for i in range(0, num_mb_per_epoch):
            
            training_x = training_data[0][i * minibatch_size:(i + 1) * minibatch_size]
            training_y = training_data[1][i * minibatch_size:(i + 1) * minibatch_size]

            if training_x.shape[0] == 0:
                break

            trainer.train_minibatch({x: training_x, y: training_y})

        # Measure training loss
        training_loss = measure_loss(training_data[0], training_data[1], x, y, trainer, minibatch_size)
        training_loss_list.append(training_loss)
        
        # Measure test loss
        validation_loss = measure_loss(validation_data[0], validation_data[1], x, y, trainer, minibatch_size)
        validation_loss_list.append(validation_loss)
        
        # Log training status
        print("epoch #{}: loss={}, val_error={}".format(e+1, training_loss_list[-1], validation_loss_list[-1]))
        trainer.save_checkpoint(checkpoint_file)

    return trainer, training_loss_list, validation_loss_list

if __name__ == '__main__':
    shape = (1, 128, 128)
    data_size = 500

    input_images, target_masks = simulation.generate_random_data(shape[1], shape[2], data_size)

    trainer, training_loss_list, validation_loss_list = train(input_images, target_masks, False)
    helper.plot_errors({"training": training_loss_list, "test": validation_loss_list}, title="Simulation Learning Curve")

