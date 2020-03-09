import subprocess


scripts = []
# scripts.append('src/eval/test_IMU_bias_sensitivity')
scripts.append('src/eval/test_IMU_sensor_sensitivity')
evalType = 'test'
numExecute = 1

for script in scripts:
    for i in range(numExecute):
        subprocess.call(f'python ./{script}.py %s' % evalType, shell=True)