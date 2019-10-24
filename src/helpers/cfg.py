import yaml
import os
from yaml import Loader as loader, Dumper as dumper
import glob
import numpy as np
# import sys
# from src.helpers.DictRef import DictRef




class ThesisConfig:
    def __init__(self, configName=None):
        self.delimiter = '.'
        if (configName is None) or (configName=='--mode=server'):
            configName = 'config/experiment.yaml'
        self.experiment = self.getCfgDict(configName)
        thesisCfgName = self.experiment['experiment']['thesisCfg']
        self.thesis = self.getCfgDict(thesisCfgName)
        print('Loaded Config from %s' % configName)

        self.loadConfig()
        self.createPathShortcuts()
        self.createParmShortcuts()


    def getCfgDict(self, pathname):
        with open(pathname, 'r') as yamlFile:
            return yaml.load(yamlFile, Loader=loader)

    def joinKeys(self, keysList):
        return self.delimiter.join(keysList)

    def splitKeys(self, refStr, maxsplits=-1):
        return refStr.split(self.delimiter, maxsplits)

    def rsplitKeys(self, refStr, maxsplits=-1):
        return refStr.rsplit(self.delimiter, maxsplits)


    def __getitem__(self, keyStr):
        cfgDictName, key = self.splitKeys(keyStr)
        cfgDict = self.__getattribute__(cfgDictName)
        return cfgDict[key]

    def loadConfig(self):
        datasets = self.thesis['datasets']
        if 'kitti' in datasets:
            kittiRef = datasets['kitti']
            kittiRef['paths']['original'] = self.getFilenamesFromYamlDict(kittiRef['paths']['original'])
            kittiRef['paths']['prepared'] = self.getFilenamesFromYamlDict(kittiRef['paths']['prepared'])
            kittiRef['paths']['normalized'] = self.getFilenamesFromYamlDict(kittiRef['paths']['normalized'])

        trainPaths = self.experiment['trainPaths']
        trainPaths = self.fillInThesisReference(trainPaths, self.thesis)
        trainPaths = self.getFilenamesFromYamlDict(trainPaths)
        self.experiment['trainPaths'] = trainPaths

    def createPathShortcuts(self):
        # Thesis
        self.kitti = self.thesis['datasets']['kitti'] # Group
        self.kittiPaths = self.kitti['paths'] # Group
        self.kittiOriginal = self.kittiPaths['original'] # File Group
        self.kittiPrepared = self.kittiPaths['prepared'] # File Group
        self.kittiNormalized = self.kittiPaths['normalized'] # File Group

        # Experiment
        self.trainPaths = self.experiment['trainPaths']

    def createParmShortcuts(self):
        # Thesis
        self.thesisKittiParms = self.kitti['parameters']  # Parm Group
        self.kittiCams = np.array(self.kittiPaths['general']['kittiOrigCameras']) # PARAMETER
        self.kittiSeqs = np.array(self.kittiPaths['general']['kittiSeqs'])  # PARAMETER
        self.splitFracs = np.array(self.thesisKittiParms['splitFractions']) # PARAMETER

        # Experiment
        self.expParms = self.experiment['parameters'] # Parm Group
        self.expKittiParms = self.expParms['kitti'] # Parm Group
        self.usedCams = np.array(self.expKittiParms['usedCams']) # PARAMETER
        self.usedSeqs = np.array(self.expKittiParms['usedSeqs']) # PARAMETER

        # training
        self.trainingParms = self.expParms['training'] # Parm Group
        self.callbackParms = self.trainingParms['callbacks']
        self.checkpointParms = self.callbackParms['checkpoint']
        self.earlyStoppingParms = self.callbackParms['earlyStopping']
        self.reduceLRParms = self.callbackParms['reduceLR']
        self.modelParms = self.expParms['model']  # Parm Group
        self.plotHistoryParms = self.callbackParms['plotHistory']
        self.constraintParms = self.trainingParms['constraints']
        self.normalizationParms = self.trainingParms['normalization']


    def expandFolders(self, directory):
        if type(directory) is str:
            return directory

        folders = []
        for part in directory:
            if type(part) is list:
                newFolders = []
                for fldr in folders:
                    for element in part:
                        newFolders.append(os.path.join(fldr, element))
                folders = newFolders
            else:
                if len(folders) > 0:
                    for i, fldr in enumerate(folders):
                        folders[i] = os.path.join(fldr, part)
                else:
                    folders.append(part)

        return folders


    def getFilenames(self, dictionary):
        returnDict = dictionary.copy()
        folders = self.expandFolders(dictionary['dir'])
        if type(folders) is str:
            folders = [folders]
        listOfFiles = []
        for fldr in folders:
            files = []
            if 'type' in dictionary:
                files = sorted(glob.glob(os.path.join(fldr, '*' + dictionary['type'])))
                if files:
                    files = [[file] for file in files]
                    retVals = {'files': np.array(files)}
                else:
                    files = [fldr]
                    retVals = {'folder': np.array(files)}

            elif 'name' in dictionary:
                files = [os.path.join(fldr, dictionary['name'])]
                retVals = {'files': np.array(files)}
            # if not files:
            # print('No such folder: %s' % fldr)
            # listOfFiles.append(np.array(files))
            listOfFiles.append(retVals)
        if len(listOfFiles) == 1:
            listOfFiles = listOfFiles[0]
        # return np.array(listOfFiles)
        returnDict['fileList'] = listOfFiles
        return returnDict


    def getFilenamesFromYamlDict(self, currLevel, saveDict=None, prevKey='', isList=False, **kwargs):
        if saveDict is None:
            saveDict = {}
        if 'dir' in currLevel:
            # print(currLevel)
            filenames = self.getFilenames(currLevel)
            # if 'usedSeqs' in kwargs:
            #     filenames = filenames[kwargs['usedSeqs']]
            saveDict[prevKey] = filenames
            return (saveDict)

        elif (isList):
            listVals = []
            for i in range(len(currLevel)):
                val = self.getFilenamesFromYamlDict(currLevel[i], saveDict, prevKey, isList=False, **kwargs)
                listVals.append(val[prevKey])
            saveDict[prevKey] = listVals
            return (saveDict)

        else:
            if type(currLevel) is list:
                saveDict = self.getFilenamesFromYamlDict(currLevel, saveDict, prevKey, isList=True, **kwargs)
            else:
                for key in currLevel:
                    saveDict = self.getFilenamesFromYamlDict(currLevel[key], saveDict, key, isList=False, **kwargs)
            return saveDict


    def fillInThesisReference(self, dictionary, thesisDict):
        for key in dictionary:
            dir = dictionary[key]['dir']
            for idx, name in enumerate(dir):
                phrase = 'thesis.'
                if phrase in name[:len(phrase)]:
                    refs = name[len(phrase):].split('.')
                    result = thesisDict
                    for ref in refs:
                        result = result[ref]
                    dir[idx] = result
        return dictionary




    def getInputFiles(self, filesDict, seq=None, cam=None):
        filesList = filesDict['fileList']
        if cam is not None:
            filesList = filesList[cam::len(self.kittiCams)]
        seqFiles = None

        if isinstance(filesList, list):
            seqFiles = filesList
            if seq is not None:
                seqFiles = seqFiles[seq]
            seqFiles = seqFiles.get('files', None)
        elif isinstance(filesList, dict):
            seqFiles = filesList.get('files', None)
            if (seq is not None) and (seqFiles is not None):
                seqFiles = seqFiles[seq]
        else:
            print('Bad Dictionary')

        if seqFiles is not None:
            if len(seqFiles.shape) == 2:
                seqFiles = seqFiles[:,0]

            for file in seqFiles:
                if not os.path.exists(file):
                    seqFiles = None
                    break

        if seqFiles is not None:
            if len(seqFiles) == 1:
                seqFiles = seqFiles[0]
        else:
            print('%sInput files do not exist. Skipping...' % (' '*4))

        return seqFiles


    def getOutputFiles(self, filesDict, overwrite=False, seq=None, cam=None):
        filesList = filesDict['fileList']
        if cam is not None:
            filesList = filesList[cam::len(self.kittiCams)]
        seqFiles = None

        if isinstance(filesList, list):
            seqFiles = filesList
            if seq is not None:
                seqFiles = seqFiles[seq]
            if 'files' in seqFiles:
                seqFiles = seqFiles['files']
            elif 'folder' in seqFiles:
                seqFiles = seqFiles['folder']
            else:
                seqFiles = None
        elif isinstance(filesList, dict):
            seqFiles = filesList
            if 'files' in seqFiles:
                seqFiles = seqFiles['files']
            elif 'folder' in seqFiles:
                seqFiles = seqFiles['folder']
            else:
                seqFiles = None
            if (seq is not None) and (seqFiles is not None):
                seqFiles = seqFiles[seq]
        else:
            print('Bad Dictionary')

        if seqFiles is not None:
            if len(seqFiles.shape) == 2:
                seqFiles = seqFiles[:, 0]

            skipMsg = False
            overwriteMsg = False
            for file in seqFiles:
                if os.path.exists(file) and (not overwrite):
                    seqFiles = None
                    skipMsg = True
                    break
                else:
                    # Else if overwrite or truthfiles do not exist
                    # create folder, if needed
                    folder = self.getFolderRef(file)

                    if self.isFileRef(file):
                        if (overwrite) and os.path.exists(file):
                            overwriteMsg = True

                    if not os.path.exists(folder):
                        os.makedirs(folder)
                        # os.chmod(folder, 0o666)

            if skipMsg:
                msg = '%sOutput File Exists. Skipping...' % (' ' * 4)
                print(msg)
            elif overwriteMsg:
                msg = '%sOutput File Exists. Overwrite...' % (' ' * 4)
                print(msg)

        if seqFiles is not None:
            if len(seqFiles) == 1:
                seqFiles = seqFiles[0]

        return seqFiles


    def isFileRef(self, path):
        return ('.' in path)

    def getFolderRef(self, filesArray):
        path = filesArray
        for i in range(len(filesArray.shape)):
            path = path[0]
        if self.isFileRef(path):
            path = os.path.dirname(path)
        return path


