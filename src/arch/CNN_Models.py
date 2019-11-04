from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Concatenate, \
    Input, Conv2D, MaxPooling2D, Dropout
from keras.applications.inception_v3 import InceptionV3




def buildCnnModel_1(image_shape):
    cnn_input = Input(shape=image_shape)
    cnn_internal = cnn_input
    cnn_internal = Conv2D(64, (3, 3), activation='relu')(cnn_internal)
    cnn_internal = MaxPooling2D((2, 2))(cnn_internal)
    cnn_internal = Conv2D(64, (3, 3), activation='relu')(cnn_internal)
    cnn_internal = MaxPooling2D((2, 2))(cnn_internal)
    cnn_internal = Conv2D(64, (3, 3), activation='relu')(cnn_internal)
    cnn_internal = MaxPooling2D((2, 2))(cnn_internal)
    cnn_output = cnn_internal
    cnn_model = Model(inputs=[cnn_input], outputs=[cnn_output], name='CNN_Model')
    return cnn_model


def buildCnnModel_2(image_shape, dropout=None):
    cnn_input = Input(shape=image_shape)
    cnn_internal = cnn_input
    cnn_internal = Conv2D(64, (3, 3), activation='relu')(cnn_internal)
    cnn_internal = MaxPooling2D((2, 2))(cnn_internal)
    cnn_internal = Conv2D(128, (3, 3), activation='relu')(cnn_internal)
    cnn_internal = MaxPooling2D((2, 2))(cnn_internal)
    if not dropout is None:
        cnn_internal = Dropout(dropout)(cnn_internal)
    cnn_internal = Conv2D(256, (3, 3), activation='relu')(cnn_internal)
    cnn_internal = MaxPooling2D((2, 2))(cnn_internal)
    cnn_internal = Conv2D(64, (3, 3), activation='relu')(cnn_internal)
    cnn_internal = MaxPooling2D((2, 2))(cnn_internal)
    if not dropout is None:
        cnn_internal = Dropout(dropout)(cnn_internal)
    cnn_output = cnn_internal
    cnn_model = Model(inputs=[cnn_input], outputs=[cnn_output], name='CNN_Model')
    return cnn_model

def buildFlowNet(image_shape):   # Should use:  64*19 x 64*5 == (1216x320)
    cnn_input = Input(shape=image_shape) # (1280 x 384 x 6)
    cnn_internal = cnn_input
    cnn_internal = Conv2D(64,   (7,7), strides=2, padding='same', activation='relu')(cnn_internal)
    cnn_internal = Conv2D(128,  (5,5), strides=2, padding='same', activation='relu')(cnn_internal)
    cnn_internal = Conv2D(256,  (5,5), strides=2, padding='same', activation='relu')(cnn_internal)
    cnn_internal = Conv2D(256,  (3,3), strides=1, padding='same', activation='relu')(cnn_internal)
    cnn_internal = Conv2D(512,  (3,3), strides=2, padding='same', activation='relu')(cnn_internal)
    cnn_internal = Conv2D(512,  (3,3), strides=1, padding='same', activation='relu')(cnn_internal)
    cnn_internal = Conv2D(512,  (3,3), strides=2, padding='same', activation='relu')(cnn_internal)
    cnn_internal = Conv2D(512,  (3,3), strides=1, padding='same', activation='relu')(cnn_internal)
    cnn_internal = Conv2D(1024, (3,3), strides=2, padding='same')(cnn_internal)
    cnn_output = cnn_internal
    cnn_model = Model(inputs=[cnn_input], outputs=[cnn_output], name='FlowNet')
    return cnn_model



# def buildFlowNet_half(image_shape):  # Should use: 32*38 x 32*11 == (1216x352)
#     cnn_input = Input(shape=image_shape) # (1280 x 384 x 3)
#     cnn_internal = cnn_input
#     cnn_internal = Conv2D(32,  (7,7), strides=2, padding='same', activation='relu')(cnn_internal)
#     cnn_internal = Conv2D(64,  (5,5), strides=2, padding='same', activation='relu')(cnn_internal)
#     cnn_internal = Conv2D(128, (5,5), strides=2, padding='same', activation='relu')(cnn_internal)
#     cnn_internal = Conv2D(128, (3,3), strides=1, padding='same', activation='relu')(cnn_internal)
#     cnn_internal = Conv2D(256, (3,3), strides=2, padding='same', activation='relu')(cnn_internal)
#     cnn_internal = Conv2D(256, (3,3), strides=1, padding='same', activation='relu')(cnn_internal)
#     cnn_internal = Conv2D(256, (3,3), strides=2, padding='same', activation='relu')(cnn_internal)
#     cnn_internal = Conv2D(256, (3,3), strides=1, padding='same', activation='relu')(cnn_internal)
#     cnn_internal = Conv2D(512, (3,3), strides=2, padding='same')(cnn_internal)
#     cnn_output = cnn_internal
#     cnn_model = Model(inputs=[cnn_input], outputs=[cnn_output], name='FlowNet_half')
#     return cnn_model


def buildInceptionV3(image_shape):
    return InceptionV3(input_shape=image_shape, include_top=False, weights='imagenet', pooling=None)



def getCnnModel(image_shape, cnn_type=None, dropout=None):
    if cnn_type == 'Inception':
        return buildInceptionV3(image_shape)
    elif cnn_type == 'WatsonCNN_1':
        return buildCnnModel_1(image_shape)
    elif cnn_type == 'WatsonCNN_2':
        return buildCnnModel_2(image_shape, dropout)
    elif cnn_type == 'FlowNet':
        return buildFlowNet(image_shape)
    elif cnn_type == 'FlowNet_half':
        return buildFlowNet_half(image_shape)
    else:
        return buildCnnModel_1(image_shape)