import subprocess
import sys
from src.helpers.cfg import ThesisConfig


# Parameters
configFile = None
if len(sys.argv)>1:
    configFile = sys.argv[1]

callStr = 'python ./{script:}.py'
if configFile is not None:
    callStr += f' {configFile:}'


scripts = ['src/prep/CreateTruthFiles',
           'src/prep/CreateSimulatedIMUFiles',
           'src/prep/StandardizeImages',
           'src/prep/DownAndNormImages',
           'src/prep/SplitKittiData',
           'src/prep/CalculateNormParms',
           'src/prep/NormalizeData']



for script in scripts:
    subprocess.call(callStr.format(script=script), shell=True)