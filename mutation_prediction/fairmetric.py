import numpy as np
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from fairness_utils import *
import typing
from typing import List, Union
class Metrics():
    def __init__(self,
                predictions:np.ndarray = None,
                probs:np.ndarray=None,
                labels:np.ndarray = None, 
                sensitives:np.ndarray = None, 
                projectName:str = "", 
                sensitiveGroupNames:list = None, 
                savePath = "", 
                saveConfusionMatrix:bool = False, 
                modify:bool = False, 
                verbose:bool = False):
        self.__dict__.update(locals())
        if(predictions is not None):
            if(not isinstance(predictions, np.ndarray)):
                if(not isinstance(predictions, list)):
                    raise TypeError(f'sequence predictions: expected numpy.ndarray or list, {type(predictions)} found.')
                predictions = np.array(predictions)

        if(labels is not None):
            if(not isinstance(labels, np.ndarray)):
                if(not isinstance(labels, list)):
                    raise TypeError(f'sequence labels: expected numpy.ndarray or list, {type(labels)} found.')
                labels = np.array(labels)

        if(sensitives is not None):
            if(not isinstance(sensitives, np.ndarray)):
                if(not isinstance(sensitives, list)):
                    raise TypeError(f'sequence sensitives: expected numpy.ndarray or list, {type(sensitives)} found.')
                sensitives = np.array(sensitives)
            
        # self.predictions = predictions
        # self.probs = probs
        # self.labels = labels
        # self.sensitives = sensitives
        # self.savePath = savePath
        if(self.savePath != ''):
            self.saveConfusionMatrix = True
        # self.saveConfusionMatrix = saveConfusionMatrix
        self.lsConfusionMetrics = []
        self.lsSensitves = []
        # self.verbose = verbose
        self.lsMetrics = []
        self.projectName = projectName
        self.dictResults = None
        self.sensitiveGroupNames = sensitiveGroupNames
        self.lsMinMaxGroups = []
        self.lsGroupCount = []
        # if(modify == False and self.predictions is not None and self.probs is not None and self.labels is not None and self.sensitives is not None):
        if all([modify == False, self.predictions is not None, self.probs is not None, self.labels is not None, self.sensitives is not None]):
            self.dictResults = self.FairnessMetrics()

    def update(self, predictions = None, probs=None, labels = None, sensitives = None, savePath = "", projectName = ""):
        if(predictions is not None):
            if(not isinstance(predictions, np.ndarray)):
                if(not isinstance(predictions, list)):
                    raise TypeError(f'sequence predictions: expected numpy.ndarray or list, {type(predictions)} found.')
                predictions = np.array(predictions)
        if (probs is not None):
            if(not isinstance(probs, np.ndarray)):
                if(not isinstance(probs, list)):
                    raise TypeError(f'sequence probs: expected numpy.ndarray or list, {type(probs)} found.')
                probs = np.array(probs)

        if(labels is not None):
            if(not isinstance(labels, np.ndarray)):
                if(not isinstance(labels, list)):
                    raise TypeError(f'sequence labels: expected numpy.ndarray or list, {type(labels)} found.')
                labels = np.array(labels)

        if(sensitives is not None):
            if(not isinstance(sensitives, np.ndarray)):
                if(not isinstance(sensitives, list)):
                    raise TypeError(f'sequence sensitives: expected numpy.ndarray or list, {type(sensitives)} found.')
                sensitives = np.array(sensitives)

        if(predictions is None): return
        self.predictions = predictions
        self.labels = labels
        if(sensitives is None): return
        self.sensitives = sensitives
        if(savePath != ""): self.savePath = savePath
        if(projectName != ""): self.projectName = projectName
        self.dictResults = self.FairnessMetrics()

        # return self.dictResults

    def FairnessRule(self):
        return FairnessMetrics(self.predictions, self.probs, self.labels, self.sensitives)

    def FairnessMetrics(self):
        # Two way to calculate fairness metrics
        # 1. init an object and FairnessMetrics()
        # 2. Metrics().FairnessMetrics(...)
        if(self.predictions is None):
            raise ValueError("missing predictions.")            
        if(self.labels is None):
            raise ValueError("missing lables.")
        if(self.sensitives is None):
            raise ValueError("missing sensitives.")
        if(self.saveConfusionMatrix == True and self.savePath == ''):
            raise ValueError("missing save confusion matrix png path.")
        
        self.dictResults = self.FairnessRule()

        lenMultiContains = [isinstance(self.dictResults[i], list) for i in self.dictResults.keys()]
        dfMulti = pd.DataFrame.from_dict(self.dictResults)
        self.lsMetrics = list(dfMulti[np.array(list(self.dictResults.keys()))[(~np.array(lenMultiContains))].tolist()].columns)
        if(self.saveConfusionMatrix):
            self.saveConfusionMatrix()
        if(self.verbose):
            print(self.__str__())
            
        return self.dictResults
    
    def getHeads(self):
        result = ""
        if(len(self.lsMetrics) == 0): return ""
        if(self.sensitiveGroupNames is None): ssg = range(len(self.dictResults['TOTALACC']))
        else: ssg = self.sensitiveGroupNames
        maxPrecision = max([len(i) for i in self.lsMetrics]+[len(f'Group(M) {i}') for i in ssg])
        # if(len(np.unique(self.sensitives)) == 2):
        if(True):
            # result = f"|{'Project':{maxPrecision}}|{'|'.join([f'{i:{maxPrecision}}' for i in self.lsMetrics[:-1]])}|"
            # result += f"{'Group1':{maxPrecision}}|{self.lsMetrics[-1]:{maxPrecision}}|{'Group2':{maxPrecision}}|"
            result = f"|{'Project':{maxPrecision}}|{'|'.join([f'{i:{maxPrecision}}' for i in self.lsMetrics[:]])}|"
            for ssgidx, i in enumerate(ssg):
                result += f"{f'Group{self.lsMinMaxGroups[ssgidx]} {i}':{maxPrecision}}|"
            result += '\n|'+'|'.join(['-'*maxPrecision for _ in range(len(self.lsMetrics)+len(self.dictResults['TOTALACC'])+1)])+'|'
        return result
    
    def getResults(self, markdownFormat = False):
        if(markdownFormat == False):
            return self.dictResults
        result = ""
        if(self.sensitiveGroupNames is None): ssg = range(len(self.dictResults['TOTALACC']))
        else: ssg = self.sensitiveGroupNames
        maxPrecision = max([len(i) for i in self.lsMetrics]+[len(f'Group(M) {i}') for i in ssg])
        # if(len(np.unique(self.sensitives)) == 2):
        if(True):
            if(len(self.lsMetrics) == 0): return ""

            # result = f"|{self.projectName:{maxPrecision}}|{'|'.join([f'{self.dictResults[i]:{maxPrecision}.4f}' for i in self.lsMetrics[:-1]])}|"
            # result += f"{self.dictResults['TOTALACC'][0]:{maxPrecision}.4f}|{self.dictResults[self.lsMetrics[-1]]:{maxPrecision}.4f}|{self.dictResults['TOTALACC'][1]:{maxPrecision}.4f}|"
            result = f"|{self.projectName:{maxPrecision}}|{'|'.join([f'{self.dictResults[i]:{maxPrecision}.4f}' for i in self.lsMetrics[:]])}|"
            result += '|'.join([f"{self.dictResults['TOTALACC'][i]:{maxPrecision}.4}" for i in range(len(self.dictResults['TOTALACC']))])+"|"
        return result
    
    def saveConfusionMatrix(self):
        if(len(self.lsSensitves) != len(self.lsConfusionMetrics)):
            self.lsSensitves = [i for i in range(len(self.lsConfusionMetrics))]
        for idx, confusionMatrix in enumerate(self.lsConfusionMetrics):
            _, _ = plot_confusion_matrix(conf_mat = confusionMatrix)
            plt.savefig(f'{self.savePath}_{self.lsSensitves[idx]}.png')
            plt.close()
    
    def __str__(self):
        if(self.dictResults is None): return ""
        msg = ''
        labels    = self.labels
        sensitives  = self.sensitives
        numUniLabels = len(np.unique(labels))
        numUniSens = len(np.unique(sensitives))+1
        
        msg += "   |"
        for i in range(numUniSens):
            for j in range(numUniLabels):
                msg += f"{j:4}|"
            if(numUniSens > i+1):
                msg += f"{' '*2}|"
        msg += "\n"
        # msg += "\n"+"-"*(11*(numUniSens-1)+4*(1+numUniLabels*numUniSens))+"\n"

        for i in range(numUniLabels):
            msg += f"{i:3}|"
            for j in range(numUniSens):
                for k in range(numUniLabels):
                    msg += f"{self.lsConfusionMetrics[j][i][k]:4}|"
                if(numUniSens > j+1):
                    msg += f"{' '*2}|"
            msg += "\n"
        lenMultiContains = [isinstance(self.dictResults[i], list) for i in self.dictResults.keys()]
        dfMulti = pd.DataFrame.from_dict(self.dictResults)

        dfSingle = dfMulti[np.array(list(self.dictResults.keys()))[(~np.array(lenMultiContains))].tolist()].iloc[:1]
        dfMulti = dfMulti[np.array(list(self.dictResults.keys()))[lenMultiContains].tolist()]
        msg += dfMulti.to_string()
        msg += "\n"
        msg += dfSingle.to_string()
        return msg
    
    def __repr__(self):
        return f'Metrics(predictions = {self.predictions}, labels = {self.labels}, sensitives = {self.sensitives})'
    
def showMetrics(results):
    precision = [len(i[0]) for i in results.items()]
    table = "| Setting | " + " | ".join(results.keys()) + " |\n"
    table += "| -----" + " | -----".join(["-"] * len(results.keys())) + "|\n| proposed |"
    for result in results.items():
        try:
            row = f" {result[1]:.4f}" + " |"  
        except ValueError:
            row = f" {result[1]}" + " |"
        table += row
    row += " \n"
    return table