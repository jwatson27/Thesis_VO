TEST=scale_test_4
REMOTE_IP=10.202.0.12
# LAST_EPOCH=200

LOCAL_USER=jwatson27
LOCAL_HOME=/home/$LOCAL_USER
LOCAL_PROJ=$LOCAL_HOME/PycharmProjects/Thesis_VO

REMOTE_USER=jwatson
REMOTE_HOME=/home/$REMOTE_USER
REMOTE_LOGIN=$REMOTE_USER@$REMOTE_IP

TRAINING_DIR=training/$TEST
EPOCHS_DIR=$TRAINING_DIR/training_epochs

#mkdir $LOCAL_HOME/thesis/$TRAINING_DIR
#scp -r $REMOTE_LOGIN:$REMOTE_HOME/$TRAINING_DIR/history $LOCAL_HOME/thesis/$TRAINING_DIR
#scp -r $REMOTE_LOGIN:$REMOTE_HOME/$TRAINING_DIR/model $LOCAL_HOME/thesis/$TRAINING_DIR

mkdir $LOCAL_HOME/thesis/$EPOCHS_DIR
scp -r $REMOTE_LOGIN:$REMOTE_HOME/$EPOCHS_DIR/epoch_*.h5 $LOCAL_HOME/thesis/$EPOCHS_DIR

#BEST_EPOCH=168
#scp -r $REMOTE_LOGIN:$REMOTE_HOME/$EPOCHS_DIR/epoch_$BEST_EPOCH.h5 $LOCAL_HOME/thesis/$EPOCHS_DIR


IN_PROGRESS: # trans_test_3: training_epochs(075-200)
# scale_test_3: training_epochs(075-200)


CDN_BACKUP_FOLDER=$LOCAL_HOME/CDN_Files

FOLDER_NAME=training
TEST=scale_test_3
SUB_FOLDER=training_epochs
LOCAL_FOLDER=$CDN_BACKUP_FOLDER/$FOLDER_NAME
mkdir $LOCAL_FOLDER
LOCAL_FOLDER=$LOCAL_FOLDER/$TEST
mkdir $LOCAL_FOLDER
LOCAL_FOLDER=$LOCAL_FOLDER/$SUB_FOLDER
mkdir $LOCAL_FOLDER
scp -r $REMOTE_LOGIN:$REMOTE_HOME/$FOLDER_NAME/$TEST/$SUB_FOLDER/* $LOCAL_FOLDER