# script path: /home/jwatson/src/train_cdn/TrainThesis_cdn.py
# Environment Variables:
#   PYTHONUNBUFFERED=1;
#   DISPLAY=unix:1
# working directory: '/home/jwatson'
# python interpreter: jwatson/thesis_env12_fixed

# Docker settings:
#   Volume Bindings (Host:Container):
#     /home/jwatson:/opt/project
#     /home/jwatson/clean_dataset:/clean_dataset
#   Environment Variables
#     NVIDIA_VISIBLE_DEVICES=0

# -v /home/jwatson:/opt/project -v /home/jwatson/clean_dataset:/clean_dataset -e NVIDIA_VISIBLE_DEVICES=0

ssh -t jwatson@10.202.0.12 bash -l

# PERSONAL COMPUTER
screen -S personal_0 bash
docker container restart docker_thesis_personal
docker run -dit --name docker_thesis_personal \
    -w '/opt/project/' \
    -v /home/jwatson27/PycharmProjects/Thesis_VO/:/opt/project \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /mnt/01D4B53BFED3AC50/Users/jwats:/windowsroot \
    -v /home/jwatson27/thesis/clean_dataset/:/clean_dataset \
    -v /mnt/01D4B53BFED3AC50/thesis/clean_dataset/:/clean_nvme \
    -v /home/jwatson27/thesis/results/:/opt/project/results \
    -v /home/jwatson27/thesis/training/:/opt/project/training \
    -v /home/jwatson27/thesis/exp_configs/:/opt/project/exp_configs \
    -e DISPLAY=unix:1 \
    -e PYTHONPATH=/opt/project/:$PYTHONPATH \
    dockernvidia418_deep-learning-with-python_service
#docker stop docker_thesis_personal
#docker rm docker_thesis_personal

docker exec -it docker_thesis_personal python ./src/prep/CreateEpipolarFiles.py

docker exec -it docker_thesis_personal python ./src/eval/RunSensitivityTest.py

docker exec -it docker_thesis_personal python ./src/eval/get_val_preds_all_epochs.py


# CDN COMPUTERS

screen -S thesis_env12_0 bash
sudo docker run -w '/opt/project' -v /home/jwatson:/opt/project -v /home/jwatson/clean_dataset:/clean_dataset -e NVIDIA_VISIBLE_DEVICES=0 -e DISPLAY=unix:1 -d --name thesis_env12_0 jwatson/thesis_env12_fixed
#sudo docker exec -it thesis_env12_0 python ./src/train_cdn/TrainThesis_cdn.py exp_configs/CNN_test_17.yaml
#sudo docker exec -it thesis_env12_0 python ./src/train_cdn/RunMultipleExperiments_cdn.py exp_configs/CNN_test_17.yaml exp_configs/CNN_test_17.yaml
#sudo docker stop thesis_env12_0
#sudo docker rm thesis_env12_0




screen -S thesis_env12_1 bash
sudo docker run -w '/opt/project' -v /home/jwatson:/opt/project -v /home/jwatson/clean_dataset:/clean_dataset -e NVIDIA_VISIBLE_DEVICES=1 -e DISPLAY=unix:1 -d --name thesis_env12_1 jwatson/thesis_env12_fixed
#sudo docker exec -it thesis_env12_1 python ./src/train_cdn/TrainThesis_cdn.py exp_configs/CNN_test_17.yaml
#sudo docker stop thesis_env12_1
#sudo docker rm thesis_env12_1




screen -S thesis_env12_2 bash
sudo docker run -w '/opt/project' -v /home/jwatson:/opt/project -v /home/jwatson/clean_dataset:/clean_dataset -e NVIDIA_VISIBLE_DEVICES=2 -e DISPLAY=unix:1 -d --name thesis_env12_2 jwatson/thesis_env12_fixed
#sudo docker exec -it thesis_env12_2 python ./src/train_cdn/TrainThesis_cdn.py exp_configs/CNN_test_17.yaml
#sudo docker stop thesis_env12_2
#sudo docker rm thesis_env12_2




screen -S thesis_env12_3 bash
sudo docker run -w '/opt/project' -v /home/jwatson:/opt/project -v /home/jwatson/clean_dataset:/clean_dataset -e NVIDIA_VISIBLE_DEVICES=3 -e DISPLAY=unix:1 -d --name thesis_env12_3 jwatson/thesis_env12_fixed
#sudo docker exec -it thesis_env12_3 python ./src/train_cdn/TrainThesis_cdn.py exp_configs/CNN_test_17.yaml
#sudo docker stop thesis_env12_3
#sudo docker rm thesis_env12_3





# DONE
# COPIED: exp_configs/trans_test_4.yaml
# COPIED: exp_configs/scale_test_4.yaml
# COPIED: exp_configs/CNN_test_12.yaml
# COPIED: exp_configs/trans_test_2.yaml
# COPIED: exp_configs/scale_test_2.yaml
# COPIED: exp_configs/trans_test_3.yaml
# COPIED: exp_configs/CNN_test_13.yaml
# COPIED: exp_configs/scale_test_3.yaml
