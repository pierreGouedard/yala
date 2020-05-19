# -*- coding: utf-8 -*-
"""
Evaluation metric for the Higgs Boson Kaggle Competition,
as described on:
https://www.kaggle.com/c/higgs-boson/details/evaluation

@author: Joyce Noah-Vanhoukce
Created: Thu Apr 24 2014
"""

import csv
import math
import pandas as pd


def create_solution_dictionary(solution):
    """ Read solution file, return a dictionary with key EventId and value (weight,label).
    Solution file headers: EventId, Label, Weight """
    
    solnDict = {}
    df_sol = pd.read_csv(solution)
    for _, row in df_sol.iterrows():
        if row['EventId'] not in solnDict:
            solnDict[row['EventId']] = (row['Class'], row['Weight'])
    return solnDict

    
def AMS(s, b):
    """ Approximate Median Significance defined as:
        AMS = sqrt(
                2 { (s + b + b_r) log[1 + (s/(b+b_r))] - s}
              )        
    where b_r = 10, b = background, s = signal, log is natural logarithm """
    
    br = 10.0
    radicand = 2 *((s+b+br) * math.log(1.0 + s/(b+br)) -s)
    if radicand < 0:
        print('radicand is negative. Exiting')
        exit()
    else:
        return math.sqrt(radicand)


def AMS_metric(solution, submission):
    """  Prints the AMS metric value to screen.
    Solution File header: EventId, Class, Weight
    Submission File header: EventId, RankOrder, Class
    """

    # solutionDict: key=eventId, value=(label, class)
    solutionDict = create_solution_dictionary(solution)

    signal = 0.0
    background = 0.0
    df_sub = pd.read_csv(submission)
    for _, row in df_sub.iterrows():
        if row['Class'] == 's': # only events predicted to be signal are scored
            if solutionDict[row['EventId']][0] == 's':
                signal += float(solutionDict[row['EventId']][1])
            elif solutionDict[row[0]][0] == 'b':
                background += float(solutionDict[row['EventId']][1])

    print('signal = {0}, background = {1}'.format(signal, background))
    print('AMS = ' + str(AMS(signal, background)))


def embedded_ams(ax_preds, ax_targets, ax_weights):
    signal, background = 0, 0
    for i, pred in enumerate(ax_preds):
        if pred == 1:
            if ax_targets[i] == 1:
                signal += ax_weights[i]
            else:
                background += ax_weights[i]

    return AMS(signal, background)


if __name__ == "__main__":

    # enter path and file names here    
    path = ""
    solutionFile = ""
    submissionFile = ""
    
    AMS_metric(solutionFile, submissionFile)
    
    
