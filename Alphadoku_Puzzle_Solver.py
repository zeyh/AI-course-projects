'''
3/29/2019
solve 30 “Alphadoku” puzzles
if it has no solution or more than one solution. 
report if no solution exists and to confirm that a solution, if it exists, is unique.
reduce the problem (transform it) 
to a corresponding Boolean Satisfiability (SAT) problem or Constraint Satisfaction Program (CSP)
rely on the heuristics used by the SAT or CSP solver. 
'''
from __future__ import print_function
import glob  # https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
import functools
import copy
import pandas as pd
import ortools
from ortools.linear_solver import pywraplp
from ortools.constraint_solver import pywrapcp
from collections import defaultdict
from ortools.sat.python import cp_model

PATH = "p"  # file folder
LEN = 5  # lenth of the grid
SIZE = 25  # length of sudoku


def assignconstraint(varlist, model, size):
    '''
    for each var, assign constraints:
        only one val for each small grid 1*1
        unique value for each row / column / grid 5*5
    '''
    sumval = getsum(size)  # 325 #add constraint that sum to 325
    # all different column, sum to 325
    for i in range(size):
        temp = []
        for j in range(size):
            temp.append(varlist[j][i])
        model.AddAllDifferent(temp)
        model.Add(sum(temp) == sumval)

    # all different row , sum to 325
    for i in range(size):
        model.AddAllDifferent(varlist[i])
        model.Add(sum(varlist[i]) == sumval)

    # all different grid, sum to 325
    # create 2d arr list to divide the matrix index from 1-25
    grid = gridgen(size)
    # display(grid)
    for k in range(size):
        temp = []
        for i in range(size):
            for j in range(size):
                if grid[i][j] == k+1:  # if matched the grid index
                    temp.append(varlist[i][j])
        model.AddAllDifferent(temp)
        model.Add(sum(temp) == sumval)

    '''
    procedure faster_all_different(cp_variables, the_cp_model):

    add OR-tools' built-in all_different constraint over cp_variables to the_cp_model 

    for each distinct pair of variables x and y in some_cp_variables:

        add a constraint to the_cp_model that x != y

    add a constraint to the_cp_model that the sum of cp_variables must be the sum of the values that correspond to letters A-Y
    '''


def solve(varlist, model, size):
    '''
    # https://developers.google.com/optimization/cp/cp_solver#maximize
    UNKNOWN:  0
    MODEL_INVALID:  1
    FEASIBLE:  2
    INFEASIBLE:  3
    OPTIMAL:  4
    '''
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    

    print("#STATUS: ",status)

    if(status == cp_model.FEASIBLE or status == cp_model.OPTIMAL):
        # retrive the value for each var assigned
        out = []
        for i in range(size):
            temp = []
            for j in range(size):
                temp.append(solver.Value(varlist[i][j]))
            out.append(temp)
        return out

    elif(status != cp_model.INFEASIBLE):
        print("local search??")
        r = model.solveLocal()
        print(r)
    # elif(status == cp_model.MODEL_INVALID or status == cp_model.UNKNOWN):
    #     print("MODEL ERROR!!!!")
    else:
        return []

    # return 0


def process(m, domain):
    '''
    param:
            square matrix m - a 2d sudoku list to solve, 
    '''
    size = len(m)
    model = cp_model.CpModel()

    varlist = assignvar(m, model, size)
    # print(varlist)
    # print(len(domain))
    assignconstraint(varlist, model, size)
    # print("find....")
    rawsolution = solve(varlist, model, size)

    if(rawsolution == []):
        return []
    else:
        solution = mapchar(rawsolution, domain)
        return solution

    return []


def main():
    files = readfiles(PATH)
    filenum = len(files)
    # filenum = 1 #for debug
    for i in range(filenum):
        # i = 3 #for debug
        print("------------------------------------")
        print("#INDEX: ",i)
        rawinput = parseFiles(files[i])
        # display(rawinput)
        domain = vardomain(rawinput)
        # print(domain)
        m = maptoint(rawinput, domain)
        # display(m)

        solution = process(m, domain)
        if(solution == []):
            print("-> UNSAT")
        else:
            solution2 = process(m, domain)
            if(solution == solution2):
                print("-> Unique")
            else:
                print("-> Not unique")
            display(solution)
            print("finished...")


def numofVar(l):
    '''
    param: 
        2d list
    return: 
        int - num of distinct var
    findout the num of distinct var in 2d list
    '''
    temp_list = functools.reduce(lambda x, y: x+y, l)
    temp_list = list(set(temp_list))
    print(temp_list)
    return len(temp_list)


def readfiles(path):
    '''
    param: 
        str:the path of the folder
    return: 
        a list of all the files in that folder
    read all files end with *cnf within a folder
    '''
    path = path+"/*.txt"
    out = []
    for file in glob.glob(path):
        out.append(file)

    return out


def parseFiles(file1):
    '''
    param: 
        a txt file
    return: 
        2d list
    parse the txt file into a 2d list for further use
    '''
    with open(file1) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        # remove empty str
        content = list(filter(None, content))
        # into a matrix
        m = []
        for i in range(len(content)):
            temp = []
            for c in content[i]:
                if(c == "_"):
                    temp.append(0)
                elif(c.isalpha()):
                    temp.append(c)
            m.append(temp)
    return m


def display(l):
    '''
    param: 
        2d list
    return: 
        none
    display the 2d list in sudoku format
    '''
    for i in range(len(l)):
        if((i+4) % 5 == 4):
            print('\n'+"------"*10)
        else:
            print()
        for j in range(len(l[i])):
            print(l[i][j], end=" ")
            if(j % 5 == 4):
                print("|", end=" ")
    print('\n'+"------"*10)


def deepCopy(copyfrom):
    return copy.deepcopy(copyfrom)


def vardomain(m):
    '''
    param:
        square matrix of sudoku
    return:
        all the unique char appear in the set
    '''
    temp_list = functools.reduce(lambda x, y: x+y, m)
    temp_list = list(set(temp_list))
    temp_list.remove(0)
    temp_list = sorted(temp_list)
    return temp_list


def maptoint(m, domain):
    '''
    param:
        m - 2d sudoku matrix in char
        domain - 1d list of a sorted list of all unique char appearance
    return:
        2d sodoku matrix in int
    map the char to int based on the char index in the list
    '''
    for i in range(len(m)):
        for j in range(len(m[i])):
            if(m[i][j] != 0):
                m[i][j] = domain.index(m[i][j]) + 1

    return m


def gridgen(size):
    '''
    param: 
        an int indicate the length of the square matrix
    return:
        a 2d matrix each grid labeled with number from 1 - length of the square matrix  
    eg: size = 4
    output:
        11 11 22 22
        11 11 22 22
        33 33 44 44
        33 33 44 44
    '''
    out = []
    for i in range(int(size/LEN)):
        for r in range(LEN):
            out1 = []
            for j in range(int(size/LEN)):
                out1 = out1 + [i*LEN+j+1]*LEN
            out.append(out1)

    return out


def mapchar(raw, domain):
    '''
    param: 
        raw 2d matrix on int
        1d list indicate all the unique letter
    return:
        2d matrix replacing the int with char
    for an int solution, map to char
    '''
    out = []
    for i in range(SIZE):
        temp = []
        for j in range(SIZE):
            temp.append(domain[raw[i][j]-1])
        out.append(temp)
    return out


def getsum(val):
    '''
    param: val
        return the sum from 1 to val
    '''
    out = (1+val)*val/2
    return int(out)


def assignvar(m, model, size):
    '''
    param: 
        2d sudoku int square matrx 
        model from ortools
        size - size of the square matrix
    return:
        a 2d list filled with model's var
    assign model var to each cell of the matrix
    '''
    # create constraint var https://developers.google.com/optimization/cp/cp_solver
    varlist = []
    for i in range(size):
        templist = []
        for j in range(size):
            if m[i][j] == 0:  # not filled
                templist.append(model.NewIntVar(1, 25, 'g'))
            else:
                templist.append(model.NewIntVar(m[i][j], m[i][j], 'g'))
        varlist.append(templist)

    return varlist


main()
