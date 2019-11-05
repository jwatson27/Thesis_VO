from src.arch.CNN_Models import getCnnModel
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Concatenate, \
    Input, Conv2D, MaxPooling2D, Dropout, Flatten
import numpy as np




def buildSingle(image_shape, cnn):
    input_layer = Input(shape=image_shape)
    out = cnn(input_layer)
    return out, input_layer


def buildDual(image_shape, cnn):
    cnn_out0, input_layer0 = buildSingle(image_shape, cnn)
    cnn_out1, input_layer1 = buildSingle(image_shape, cnn)
    input_layers = [input_layer0, input_layer1]
    merged_output = Concatenate()([cnn_out0, cnn_out1])
    return merged_output, input_layers



def buildModel(image_shape, num_outputs=6, cnn_type=None,
               vo_dropout=None, cnn_dropout=None,
               include_imu=False, imu_dense_size=16,
               include_epi_rot=False, epi_rot_dense_size=16,
               include_epi_trans=False, epi_trans_dense_size=16):

    # Get CNN Architecture
    # Pooling None means output is a 4D tensor avg or max means 2D tensor
    internal_cnn = getCnnModel(image_shape, cnn_type, dropout=cnn_dropout)
    internal_cnn.summary()

    if cnn_type == 'FlowNet':
        image_shape = tuple(np.append(np.array(image_shape[:-1]), 6))
        out, input_layer = buildSingle(image_shape, internal_cnn)
        input_layers = [input_layer]
    else:
        merged_out, input_layers = buildDual(image_shape, internal_cnn)
    out = GlobalAveragePooling2D()(out)

    if include_imu:
        imu_layer_size = imu_dense_size
        imu_input = Input(shape=(3,))
        input_layers.append(imu_input)
        imu_output = Dense(imu_layer_size, activation='relu')(imu_input)
        out = Concatenate()([out, imu_output])


    if include_epi_rot:
        pass
    if include_epi_trans:
        pass

    # Dropout layer
    if vo_dropout is not None:
        out = Dropout(vo_dropout)(out)

    # Output layer
    output_final = Dense(num_outputs, activation='linear')(out)

    # Build Model
    model = Model(inputs=input_layers, outputs=[output_final])
    model.summary()

    return model, internal_cnn