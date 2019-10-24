from src.arch.CNN_Models import getCnnModel
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Concatenate, \
    Input, Conv2D, MaxPooling2D, Dropout, Flatten



# TODO: Create Build Model Function based on Config file
def buildDualHeadModel(image_shape, num_outputs=6, vo_dropout=None, cnn_dropout=None, cnn_type=None, include_imu=False):

    # Get CNN Architecture
    # Pooling None means output is a 4D tensor avg or max means 2D tensor
    internal_cnn = getCnnModel(image_shape, cnn_type, dropout=cnn_dropout)
    internal_cnn.summary()

    # Define Dual-Headed Architecture
    input_layer0 = Input(shape=image_shape)
    input_layer1 = Input(shape=image_shape)
    input_layers = [input_layer0, input_layer1]

    cnn_out0 = internal_cnn(input_layer0)
    cnn_out1 = internal_cnn(input_layer1)
    merged_output = Concatenate()([cnn_out0, cnn_out1])

    # Final Layers
    x = GlobalAveragePooling2D()(merged_output)


    if include_imu:
        imu_layer_size = 16
        imu_input = Input(shape=(3,))
        input_layers.append(imu_input)
        imu_output = Dense(imu_layer_size, activation='relu')(imu_input)
        x = Concatenate()([x, imu_output])


    x = Dense(2048, activation='relu')(x)
    if not vo_dropout is None:
        x = Dropout(vo_dropout)(x)
    output_final = Dense(num_outputs, activation='linear')(x)

    # Build Model
    model = Model(inputs=input_layers, outputs=[output_final])
    model.summary()

    return (model, internal_cnn)