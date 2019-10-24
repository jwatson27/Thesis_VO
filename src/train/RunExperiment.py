import subprocess


# Parameters
configFile = '/opt/project/exp_configs/CNN_test_0.yaml'


scripts = ['src/prep/RunPrep',
           'src/train/TrainThesis']

for script in scripts:
    subprocess.call(f'python ./{script}.py {configFile}', shell=True)