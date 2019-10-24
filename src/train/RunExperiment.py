import subprocess
from src.helpers.cfg import ThesisConfig


# Parameters
configFile = '/opt/project/exp_configs/CNN_test_0.yaml'

config = ThesisConfig(configFile)
nextEpoch = config.trainingParms['nextEpoch']
runPrepDict = config.expKittiParms['prepared']['runPrep']
runPrep = True in list(runPrepDict.values())

scripts = []
if runPrep and nextEpoch<2:
    scripts.append('src/prep/RunPrep')

scripts.append('src/train/TrainThesis')

for script in scripts:
    subprocess.call(f'python ./{script}.py {configFile}', shell=True)