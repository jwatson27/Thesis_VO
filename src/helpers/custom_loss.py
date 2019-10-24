
import keras.backend as K


def scaledMSE_RT(rotScale=1):

    def lossFunction(y_true, y_pred):

        R_true, R_pred = y_true[:3], y_pred[:3]
        R_mse = K.mean(K.square(R_true - R_pred))

        T_true, T_pred = y_true[3:], y_pred[3:]
        T_mse = K.mean(K.square(T_true - T_pred))

        return K.sum(rotScale*R_mse + T_mse)

    return lossFunction