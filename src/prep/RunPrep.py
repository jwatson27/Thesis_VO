import subprocess


# Parameters
configFile = 'config/experiment.yaml'


scripts = ['src/prep/CreateTruthFiles',
           'src/prep/CreateSimulatedIMUFiles',
           'src/prep/StandardizeImages',
           'src/prep/DownAndNormImages',
           'src/prep/SplitKittiData',
           'src/prep/CalculateNormParms',
           'src/prep/NormalizeData']

for script in scripts:
    subprocess.call('python ./%s.py %s' % (script, configFile), shell=True)