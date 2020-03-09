USER=jwatson
REMOTE=10.202.0.11:volume1/homes/$USER
LOCAL=/home/jwatson27/CDN_home/
sudo mount -t nfs $REMOTE $LOCAL