import subprocess
from src.helpers.cfg import ThesisConfig
import sys

# Parameters
configFileList = None
if len(sys.argv)>1:
    configFileList = sys.argv[1:]

    for configFile in configFileList:
        config = ThesisConfig(configFile)
        nextEpoch = config.trainingParms['nextEpoch']
        runPrepDict = config.expKittiParms['prepared']['runPrep']
        runPrep = True in list(runPrepDict.values())

        scripts = []
        # if runPrep and nextEpoch<2:
        #     scripts.append('src/prep/RunPrep')

        scripts.append('src/train_cdn/TrainThesis_cdn')

        for script in scripts:
            subprocess.call(f'python ./{script}.py {configFile}', shell=True)