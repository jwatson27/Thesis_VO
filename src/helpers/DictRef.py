from functools import reduce

class DictRef():
    def __init__(self, dictionary, reference=''):
        self.delimiter = '/'
        if isinstance(dictionary, DictRef):
            self.data = dictionary.data
            self.ref = reference
            if dictionary.ref:
                self.ref = self.joinKeys([dictionary.ref, self.ref])
        else:
            self.data = dictionary
            self.ref = reference
        self.ref_dict = self.getRefDict(self.ref, self.data)

    def __getitem__(self, keyStr):
        return self.getRefDict(keyStr)

    def __getattr__(self, keyStr):
        return DictRef(self, keyStr)

    def __setitem__(self, keyStr, value):
        reference = self.ref
        if keyStr:
            reference = self.joinKeys([reference,keyStr])
        refs = self.rsplitKeys(reference, 1)
        dref = self.getRefDict(refs[0], self.data)
        if len(refs)>1:
            curr = refs[1]
            dref[curr] = value
        self.ref_dict = self.getRefDict(self.ref, self.data)

    def getRefDict(self, reference='', dictionary=None):
        if dictionary is None:
            dictionary = self.ref_dict
        if reference:
            refs = self.splitKeys(reference)
            try:
                refDict = reduce(dict.get, refs, dictionary)
            except TypeError:
                refDict = None
            if refDict is None:
                print('Invalid Key')
        else:
            refDict = dictionary
        return refDict

    def __iter__(self):
        return iter(self.ref_dict)

    def __str__(self):
        return repr(self.ref_dict)

    def keys(self):
        return self.ref_dict.keys()

    def joinKeys(self, keysList):
        return self.delimiter.join(keysList)

    def splitKeys(self, refStr, maxsplits=-1):
        return refStr.split(self.delimiter, maxsplits)

    def rsplitKeys(self, refStr, maxsplits=-1):
        return refStr.rsplit(self.delimiter, maxsplits)






#
#
#
#
#
#
#
#
#
# # class D(MutableMapping):
# #     '''
# #     Mapping that works like both a dict and a mutable object, i.e.
# #     d = D(foo='bar')
# #     and
# #     d.foo returns 'bar'
# #     '''
# #     # ``__init__`` method required to create instance from class.
# #     def __init__(self, *args, **kwargs):
# #         '''Use the object dict'''
# #         # self.dictionary = dict()
# #         self.__dict__.update(*args, **kwargs)
# #     # The next five methods are requirements of the ABC.
# #     def __setitem__(self, key, value):
# #         self.__dict__[key] = value
# #     def __getitem__(self, key):
# #         retval = self.__dict__[key]
# #         if isinstance(retval, dict):
# #             retval = retval
# #         return retval
# #     def __delitem__(self, key):
# #         del self.__dict__[key]
# #     def __iter__(self):
# #         return iter(self.__dict__)
# #     def __len__(self):
# #         return len(self.__dict__)
# #     # The final two methods aren't required, but nice for demo purposes:
# #     def __str__(self):
# #         '''returns simple dict representation of the mapping'''
# #         return str(self.__dict__)
# #     def __repr__(self):
# #         '''echoes class, id, & reproducible representation in the REPL'''
# #         return '{}, D({})'.format(super(D, self).__repr__(),
# #                                   self.__dict__)
#
#
#
# from collections import defaultdict
#
#
# #
# # test_dict = {'a': {'b': 0, 'c': 1}}
# # get_nested_value(test_dict, 'a/d')
# #
# # test_dict.get('a').get('d')
#
#
# class ConfigDict(MutableMapping):
#     def __init__(self, data=None):
#         self.dictionary = defaultdict()
#         if isinstance(data, str):
#             data = self.getCfgDict(data)
#         self.update(data)
#     def __getitem__(self, key):
#         return self.get_nested_value(key)
#     def __delitem__(self, key):
#         value = self[key]
#         del self.dictionary[key]
#         self.pop(value, None)
#     def __setitem__(self, key, value):
#         if isinstance(value, dict):
#             value = ConfigDict(value)
#         self.dictionary[key] = value
#     def __iter__(self):
#         return iter(self.dictionary)
#     def __len__(self):
#         return len(self.dictionary)
#     def __repr__(self):
#         return repr(self.dictionary)
#     def getCfgDict(self, pathname):
#         with open(pathname, 'r') as yamlFile:
#             return yaml.load(yamlFile, Loader=loader)
#     def update(self, *args, **kwargs):
#         for k, v in dict(*args, **kwargs).items():
#             self.dictionary[k] = v
#     def get_nested_value(self, path=''):
#         path = path.split('/')
#         print(path)
#         if path:
#             return reduce(dict.get, path, self.dictionary)
#         else:
#             return self.dictionary
#
#
# class BaseDict(dict):
#     def __init__(self, *args, **kw):
#         if 'cfgPath' in kw:
#             path = kw.pop('cfgPath')
#             super(BaseDict,self).__init__(self.getCfgDict(path))
#         else:
#             super(BaseDict,self).__init__(*args, **kw)
#     # def __setitem__(self, key, value):
#         # super(BaseDict,self).__setitem__(key, value)
#     def __iter__(self):
#         return super(BaseDict,self).__iter__()
#     def keys(self):
#         return super(BaseDict,self).keys()
#     # def values(self):
#     #     return super(BaseDict,self).values()
#     def getCfgDict(self, pathname):
#         with open(pathname, 'r') as yamlFile:
#             return yaml.load(yamlFile, Loader=loader)
#
# class ConfigBaseDict(dict):
#     def __init__(self, *args, **kwargs):
#         if args and type(args[0]) is str:
#             cfgPath = args[0]
#         elif 'cfgPath' in kwargs:
#             cfgPath = kwargs.pop('path')
#         else:
#             cfgPath = None
#
#         if args and type(args[0]) is dict:
#             dictionary = args[0]
#         elif 'dict' in kwargs:
#             dictionary = kwargs.pop('dict')
#         else:
#             dictionary = {}
#
#
#         if cfgPath is not None:
#             dictionary = self.getCfgDict(cfgPath)
#         self.update(dictionary)
#
#
#     def getCfgDict(self, pathname):
#         with open(pathname, 'r') as yamlFile:
#             return yaml.load(yamlFile, Loader=loader)
#
#     def __getitem__(self, key):
#         val = dict.__getitem__(self, key)
#         # if isinstance(val, dict):
#         #     val = ConfigBaseDict(val)
#         # print('val:', val)
#         return val
#
#     def __setitem__(self, key, value):
#         print('key', 'value:', key, value)
#         dict.__setitem__(self, key, value)
#         print('self:', self)
#
#     def __repr__(self):
#         dictrepr = dict.__repr__(self)
#         # print('dictrepr:', dictrepr)
#         return dictrepr
#
#     def update(self, *args, **kwargs):
#         for k, v in dict(*args, **kwargs).items():
#             self[k] = v
#
#
#
#
#
# #
# # class ConfigDict(UserDict):
# #     def __init__(self, initialdata=None, cfgPath=None):
# #         if initialdata is not None:
# #             if type(initialdata) is dict:
# #                 self.data = initialdata
# #         if cfgPath is not None:
# #             self.data = self.getCfgDict(cfgPath)
# #
# #     def getCfgDict(self, pathname):
# #         with open(pathname, 'r') as yamlFile:
# #             return yaml.load(yamlFile, Loader=loader)
# #
# #     def __getitem__(self, reference):
# #         refs = reference.split('/')
# #         curr_level = self
# #         for ref in refs:
# #             if ref:
# #                 curr_level = UserDict.__getitem__(curr_level, ref)
# #                 if type(curr_level) is dict:
# #                     curr_level = ConfigDict(curr_level)
# #                 # curr_level = curr_level[ref]
# #         return curr_level
# #
# #     def __setitem__(self, reference, value):
# #         refs = ('/'+reference).rsplit('/',1)
# #         curr_level = self[refs[0]]
# #         UserDict.__setitem__(curr_level, refs[1], value)
#
#
#         # self.SetItem(reference, value)
#
#
#         # return self.GetItem(reference)
#
#
#
#     # def GetItem(self, reference):
#     #     refs = reference.split('/')
#     #     currDict = self.cfgDict # dict object
#     #     for i, ref in enumerate(refs):
#     #         if ref:
#     #             if ref in currDict:
#     #                 currDict = currDict[ref]
#     #             else:
#     #                 currRef = '/'.join(refs[:i])
#     #                 print('Invalid Key: \'%s\'' % ref)
#     #                 print('Possible keys for [\'%s\'] are:\n %s' % (currRef, list(currDict.keys())))
#     #                 return None
#     #
#     #     if type(currDict) is dict:
#     #         return ConfigDict(currDict)
#     #     else:
#     #         return currDict
#     #
#     #
#     # def __setitem__(self, reference, value):
#     #     self.SetItem(reference, value)
#     #
#     # def SetItem(self, reference, value):
#     #     refs = reference.split('/')
#     #     currDict = self.cfgDict
#     #     for i, ref in enumerate(refs):
#     #         if ref:
#     #             if ref in currDict:
#     #                 currDict = currDict[ref]
#     #
#     #             else:
#     #                 currRef = '/'.join(refs[:i])
#     #                 print('Invalid Key: \'%s\'' % ref)
#     #                 print('Possible keys for [\'%s\'] are:\n %s' % (currRef, list(currDict.keys())))
#     #                 return
#     #
#     #     currDict = value
#     #
#     # def __str__(self):
#     #     return yaml.dump(self.data, default_flow_style=False)
#
#     # def __repr__(self):
#     #     return str(self) #'  Keys: ' + str(list(self.keys()))
#
#     # def __iter__(self):
#     #     return self.cfgDict.__iter__()
#
#     # def keys(self):
#     #     return (self.data.keys())
#
#
#
#
#
#
# # class ConfigDict:
# #     def __init__(self, cfgDict=None, cfgPath=None):
# #         self.cfgDict = {}
# #
# #         if cfgDict is not None:
# #             self.cfgDict = cfgDict
# #         else:
# #             if cfgPath is not None:
# #                 self.cfgDict = self.getCfgDict(cfgPath)
# #
# #
# #     def getCfgDict(self, pathname):
# #         with open(pathname, 'r') as yamlFile:
# #             return yaml.load(yamlFile, Loader=loader)
# #
# #     def __getitem__(self, reference):
# #         return self.GetItem(reference)
# #
# #     def GetItem(self, reference):
# #         refs = reference.split('/')
# #         currDict = self.cfgDict # dict object
# #         for i, ref in enumerate(refs):
# #             if ref:
# #                 if ref in currDict:
# #                     currDict = currDict[ref]
# #                 else:
# #                     currRef = '/'.join(refs[:i])
# #                     print('Invalid Key: \'%s\'' % ref)
# #                     print('Possible keys for [\'%s\'] are:\n %s' % (currRef, list(currDict.keys())))
# #                     return None
# #
# #         if type(currDict) is dict:
# #             return ConfigDict(currDict)
# #         else:
# #             return currDict
# #
# #
# #     def __setitem__(self, reference, value):
# #         self.SetItem(reference, value)
# #
# #     def SetItem(self, reference, value):
# #         refs = reference.split('/')
# #         currDict = self.cfgDict
# #         for i, ref in enumerate(refs):
# #             if ref:
# #                 if ref in currDict:
# #                     currDict = currDict[ref]
# #
# #                 else:
# #                     currRef = '/'.join(refs[:i])
# #                     print('Invalid Key: \'%s\'' % ref)
# #                     print('Possible keys for [\'%s\'] are:\n %s' % (currRef, list(currDict.keys())))
# #                     return
# #
# #         currDict = value
# #
# #     def __str__(self):
# #         return yaml.dump(self.cfgDict, default_flow_style=False)
# #
# #     def __repr__(self):
# #         return '  Keys: ' + str(list(self.keys()))
# #
# #     def __iter__(self):
# #         return self.cfgDict.__iter__()
# #
# #     def keys(self):
# #         return (self.cfgDict.keys())
#
#         # if (len(refs)==1):
#         #     if refs[0] in self.cfgDict:
#         #         currVal = self.cfgDict[refs[0]]
#         #         if type(currVal) is dict:
#         #             return ConfigDict(currVal)
#         #         else:
#         #             return currVal
#         #     else:
#         #         print(
#         #         raise KeyError(refs[0])
#         # elif (len(refs)==2):
#
#
#     # def __getitem__(self, reference):
#     #     # print(reference, self.cfgDict.keys())
#     #     refs = reference.split('/', 1)
#     #
#     #     if (len(refs)>1) and (refs[1]):
#     #         # try:
#     #         next_level = ConfigDict(self.cfgDict[refs[0]])
#     #         # print('next_level')
#     #         try:
#     #         #     print(refs[1])
#     #             return next_level[refs[1]]
#     #         except KeyError as ke:
#     #             print()
#     #             print('Invalid Reference: %s' % ke)
#     #             print()
#     #             currRef = reference.rsplit('/', 1)[0]
#     #             print('Valid Keys for [\'%s\'] are:\n  %s' % (currRef, list(self.cfgDict[refs[0]].keys())))
#     #             # print()
#             # except KeyError:
#             #     currRef = '' #reference.rsplit('/', 1)[0]
#             #     print('Valid Keys for [\'%s\'] are:\n  %s' % (currRef, list(self.cfgDict.keys())))
#             #     raise
#
#             #
#             # if refs[0] in self.cfgDict:
#             #     # return ConfigDict(self.cfgDict[refs[0]])[refs[1]]
#             #
#             #     nextLevel = ConfigDict(self.cfgDict[refs[0]])[refs[1]]
#             #     if not nextLevel is None:
#             #         return nextLevel
#             #     else:
#             #         currRef = reference.rsplit('/', 1)[0]
#             #         print('Valid Keys for [\'%s\'] are:\n  %s' % (currRef, list(self.cfgDict[refs[0]].keys())))
#             #
#             # else:
#             #     print('Invalid Reference: \'%s\'' % refs[0])
#             #     return None
#
#
#
#
#
#
#     #     if refs[0] in self.cfgDict:
#     #         return self.cfgDict[refs[0]][refs[1]]
#     #     else:
#     #         print('Invalid Reference: \'%s\'' % refs[0])
#     #         # currRef = reference.split('/', 1)[0]
#     #         print('Valid Keys are:\n  %s' % (list(currVal.keys())))
#     #         return None
#     #
#     #
#     #     for ref in reference.split('/', 1):
#     #         if ref:
#     #             if ref in currVal:
#     #                 currVal = currVal[ref]
#     #             else:
#     #                 print('Invalid Reference: \'%s\'' % reference)
#     #                 # currRef = reference.split('/', 1)[0]
#     #                 print('Valid Keys are:\n  %s' % (list(currVal.keys())))
#     #                 return None
#     #
#     #     if type(currVal) is dict:
#     #         return ConfigDict(currVal)
#     #     else:
#     #         return currVal
#     #
#     # def __setitem__(self, reference, value):
#     #     currVal = self.cfgDict
#     #     for ref in reference.split('/'):
#     #         if ref:
#     #             if ref in currVal:
#     #                 currVal = currVal[ref]
#     #             else:
#     #                 print('Invalid Reference: \'%s\'' % reference)
#     #                 # currRef = reference.split('/', 1)[0]
#     #                 print('Valid Keys are:\n  %s' % (list(currVal.keys())))
#     #                 return None
#
#
#         # print(currVal)
#         # currVal = value
#         # print(currVal)
#         # print('Config Dict Set Item')
#         # # print(self.cfgDict)
#         #
#         #
#         # # currVal = self.cfgDict
#         # # for ref in reference.split('/'):
#         # #     if ref:
#         # #         if ref in currVal:
#         # #             currVal = currVal[ref]
#         # #         else:
#         # #             print('Invalid Reference: \'%s\'' % reference)
#         # #             currRef = reference.split('/', 1)[0]
#         #             # print('Valid Keys are:\n  %s' % (list(currVal.keys())))
#         #             # return None
#         #
#         # # currVal = value
#         #
#         #
#         #
#         # ref = reference.split('/', 1)
#         # cfg = self.__getattribute__(ref[0])
#         # cfg[ref[1]] = value
#
#
#
#
# #
# # class YamlConfig:
# #     def __init__(self, configName):
# #         self.configPathname = configName
# #         self.configDict = self.loadCfgDict(configName)
# #
# #     def __str__(self):
# #         return str(self.configDict)
# #
# #     def __repr__(self):
# #         return repr(self.configDict)
# #
# #     def __iter__(self):
# #         return self.configDict.__iter__()
# #
# #     # def __getattr__(self, name):
# #     #     # print('YamlConfig name: %s' % name)
# #     #     return self[str(name)]
# #
# #     def __getitem__(self, reference):
# #         try:
# #             return YamlConfig(self.configDict[reference])
# #         except KeyError as ke:
# #             print()
# #             print('Invalid Reference: %s' % ke)
# #             print()
# #             currRef = ''  # reference.rsplit('/', 1)[0]
# #             print('Valid Keys for [\'%s\'] are:\n  %s' % (currRef, list(self.configDict.keys())))
# #             # print()
# #         # retVal = self.configDict[reference]
# #         # if not retVal is None:
# #         #     return retVal
# #         # else:
# #         #     print('Valid Keys are:\n  %s' % (list(self.configDict.keys())))
# #
# #
# #     # def __setitem__(self, reference, value):
# #     #     print('Yaml Config Set Item')
# #     #     print('YAML', self.configDict[reference])
# #     #     self.configDict[reference] = value
# #     #     print(self.configDict)
# #
# #
# #
# #     def loadCfgDict(self, pathname):
# #         with open(pathname, 'r') as yamlFile:
# #             yamlStr = yaml.load(yamlFile, Loader=loader)
# #             return ConfigDict(yamlStr)
# #
# #     def keys(self):
# #         return (self.configDict.keys())