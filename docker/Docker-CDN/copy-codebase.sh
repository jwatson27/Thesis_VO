LOCAL_USER=jwatson27
LOCAL_HOME=/home/$LOCAL_USER
LOCAL_PROJ=$LOCAL_HOME/PycharmProjects/Thesis_VO

REMOTE_USER=jwatson
REMOTE_IP=10.202.0.12
REMOTE_HOME=/home/$REMOTE_USER
REMOTE_LOGIN=$REMOTE_USER@$REMOTE_IP

scp -r $LOCAL_PROJ/docker/Docker-CDN/dockerfile_jwatson $REMOTE_LOGIN:$REMOTE_HOME
scp -r $LOCAL_PROJ/src $REMOTE_LOGIN:$REMOTE_HOME
scp -r $LOCAL_PROJ/config $REMOTE_LOGIN:$REMOTE_HOME
scp -r $LOCAL_HOME/thesis/exp_configs/ $REMOTE_LOGIN:$REMOTE_HOME

#scp -r $LOCAL_PROJ/docker/Docker-CDN/docker-compose.yml $REMOTE_LOGIN:$REMOTE_HOME

scp -r $LOCAL_PROJ/src/train_cdn/RunMultipleExperiments_cdn.py $REMOTE_LOGIN:$REMOTE_HOME/src/train_cdn