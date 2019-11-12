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
               include_epi_rot=False, include_epi_trans=False,
               split_epi_layers=False, epi_dense_size=16, post_constraints_layer_size=0):

    # Get CNN Architecture
    # Pooling None means output is a 4D tensor avg or max means 2D tensor

    if cnn_type == 'FlowNet' or cnn_type=='FlowNet_half' or cnn_type=='FlowNet_quarter':
        image_shape = tuple(np.append(np.array(image_shape[:-1]), image_shape[-1]*2))
        print(image_shape)
        internal_cnn = getCnnModel(image_shape, cnn_type, dropout=cnn_dropout)
        out, input_layer = buildSingle(image_shape, internal_cnn)
        input_layers = [input_layer]
    else:
        internal_cnn = getCnnModel(image_shape, cnn_type, dropout=cnn_dropout)
        merged_out, input_layers = buildDual(image_shape, internal_cnn)
    out = GlobalAveragePooling2D()(out)
    internal_cnn.summary()

    if include_imu:
        imu_layer_size = imu_dense_size
        imu_input = Input(shape=(3,))
        input_layers.append(imu_input)
        imu_output = Dense(imu_layer_size, activation='relu')(imu_input)
        out = Concatenate()([out, imu_output])

    if include_epi_rot or include_epi_trans:
        epi_layer_size = epi_dense_size
        epi_rot_input = Input(shape=(3,))
        epi_trans_input = Input(shape=(3,))
        if include_epi_rot and include_epi_trans and not split_epi_layers:
            input_layers.append(epi_rot_input)
            input_layers.append(epi_trans_input)
            epi_input = Concatenate()([epi_rot_input, epi_trans_input])
            epi_output = Dense(epi_layer_size, activation='relu')(epi_input)
            out = Concatenate()([out, epi_output])
        else: # not both or split
            if include_epi_rot:
                input_layers.append(epi_rot_input)
                epi_rot_output = Dense(epi_layer_size, activation='relu')(epi_rot_input)
                out = Concatenate()([out, epi_rot_output])
            if include_epi_trans:
                input_layers.append(epi_trans_input)
                epi_trans_output = Dense(epi_layer_size, activation='relu')(epi_trans_input)
                out = Concatenate()([out, epi_trans_output])

    if include_imu or include_epi_rot or include_epi_trans:
        # Add extra dense layer
        if not post_constraints_layer_size==0:
            out = Dense(post_constraints_layer_size, activation='relu')(out)



    # Dropout layer
    if vo_dropout is not None:
        out = Dropout(vo_dropout)(out)

    # Output layer
    output_final = Dense(num_outputs, activation='linear')(out)

    # Build Model
    model = Model(inputs=input_layers, outputs=[output_final])
    model.summary()

    return model, internal_cnn