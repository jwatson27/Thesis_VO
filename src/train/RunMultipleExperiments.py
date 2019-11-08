import subprocess
from src.helpers.cfg import ThesisConfig


# Parameters
configFileList = ['exp_configs/CNN_test_9.yaml',
                  'exp_configs/CNN_test_6.yaml',
                  'exp_configs/CNN_test_4.yaml',
                  'exp_configs/CNN_test_10.yaml']

for configFile in configFileList:
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