LOCAL_USER=jwatson27
LOCAL_HOME=/home/$LOCAL_USER
LOCAL_PROJ=$LOCAL_HOME/PycharmProjects/Thesis_VO

REMOTE_USER=jwatson
REMOTE_IP=10.202.0.12
REMOTE_HOME=/home/$REMOTE_USER
REMOTE_LOGIN=$REMOTE_USER@$REMOTE_IP

#ssh $REMOTE_LOGIN
#mkdir -p clean_dataset/kitti_odom/00/cam_0
#mkdir -p clean_dataset/kitti_odom/01/cam_0
#mkdir -p clean_dataset/kitti_odom/02/cam_0
#mkdir -p clean_dataset/kitti_odom/03/cam_0
#mkdir -p clean_dataset/kitti_odom/04/cam_0
#mkdir -p clean_dataset/kitti_odom/05/cam_0
#mkdir -p clean_dataset/kitti_odom/06/cam_0
#mkdir -p clean_dataset/kitti_odom/07/cam_0
#mkdir -p clean_dataset/kitti_odom/08/cam_0
#mkdir -p clean_dataset/kitti_odom/09/cam_0
#mkdir -p clean_dataset/kitti_odom/10/cam_0
#exit

scp -r $LOCAL_HOME/thesis/clean_dataset/kitti_odom/split_idxs.hdf5 $REMOTE_LOGIN:$REMOTE_HOME/clean_dataset/kitti_odom

scp -r $LOCAL_HOME/thesis/clean_dataset/kitti_odom/* $REMOTE_LOGIN:$REMOTE_HOME/clean_dataset/kitti_odom

scp -r $LOCAL_HOME/thesis/clean_dataset/kitti_odom/00/imu.hdf5 $REMOTE_LOGIN:$REMOTE_HOME/clean_dataset/kitti_odom/00
scp -r $LOCAL_HOME/thesis/clean_dataset/kitti_odom/00/truth.hdf5 $REMOTE_LOGIN:$REMOTE_HOME/clean_dataset/kitti_odom/00
scp -r $LOCAL_HOME/thesis/clean_dataset/kitti_odom/00/cam_0/epipolar.hdf5 $REMOTE_LOGIN:$REMOTE_HOME/clean_dataset/kitti_odom/00/cam_0
scp -r $LOCAL_HOME/thesis/clean_dataset/kitti_odom/00/cam_0/epipolar_masked.hdf5 $REMOTE_LOGIN:$REMOTE_HOME/clean_dataset/kitti_odom/00/cam_0

KITTI_NORM=/mnt/01D4B53BFED3AC50/thesis/clean_dataset/kitti_norm
scp -r $KITTI_NORM/normalized/norm_epi.hdf5 $REMOTE_LOGIN:$REMOTE_HOME/clean_dataset/kitti_norm/normalized




TEST=CNN_test_12
LAST_EPOCH=075

TRAINING_DIR=training/$TEST
EPOCHS_DIR=$TRAINING_DIR/training_epochs

mkdir $REMOTE_LOGIN:$REMOTE_HOME/$TRAINING_DIR
mkdir $REMOTE_LOGIN:$REMOTE_HOME/$EPOCHS_DIR
scp -r $LOCAL_HOME/thesis/$EPOCHS_DIR/epoch_$LAST_EPOCH.h5 $REMOTE_LOGIN:$REMOTE_HOME/$EPOCHS_DIR
scp -r $LOCAL_HOME/thesis/$TRAINING_DIR/history $REMOTE_LOGIN:$REMOTE_HOME/$TRAINING_DIR

#scp -r $LOCAL_HOME/thesis/exp_configs/scale_test_4.yaml $REMOTE_LOGIN:$REMOTE_HOME/exp_configs
#scp -r $LOCAL_HOME/thesis/exp_configs/trans_test_4.yaml $REMOTE_LOGIN:$REMOTE_HOME/exp_configs