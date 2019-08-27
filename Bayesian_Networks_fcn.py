
import glob #https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
import functools
import copy
import random

PATH = "c"
MAXNUM = 3


#ret num of clause satisfied for a rand gen p
def evaluate(c, p):
    temp_list = functools.reduce(lambda x,y :x+y , c)
    temp_list = map(abs, temp_list) #only want the abs val - elimate -
    temp_list = list(set(temp_list))
    # print(temp_list)
    lenth = len(temp_list)
    # print(len(p), lenth)
    # print("total",lenth)
    # print("len:",len(c))
    count = 0
    # print("#len",len(p))
    for i in range(len(c)):
        temp = 0
        for j in range(len(c[i])):
            # print(abs(c[i][j]))          
            temp = temp + p[abs(c[i][j])-1]
        if(temp >= -1):
            count = count + 1       
    return count



def genBest(c,p):
    '''
    param:
            2d list cnf
            1d assign p 
    return:
            best p 
            best heuristic val
    twick each element in p to get best val
    '''
    p_1 = DeepCopy(p)
    val_1 = evaluate(c,p)
    #loop all assignment find max eval
    for i in range(len(p)):
        p[i] = -1*p[i]
        tempval = evaluate(c,p)
        if(tempval > val_1):
            val_1 = tempval
            p_1 = DeepCopy(p)
        p[i] = -1*p[i]
    # print(p_1, val_1)
    return p_1,val_1
    

def ranGen(c):
    '''
    param:
        2d list cnf
    return:
        1d list with len var num
    random assign to var
    '''
    temp_list = functools.reduce(lambda x,y :x+y , c)
    temp_list = map(abs, temp_list) #only want the abs val - elimate -
    temp_list = list(set(temp_list))
    lenth = len(temp_list)
    out = []
    for i in range(lenth):
        out.append(random.randrange(-1,2,2)) #gen random -1 or 1 size lenth
    return out


def changeformat(inlist):
    '''
    change the format
    '''
    out = []
    for i in range(len(inlist)):
        out.append((i+1)*inlist[i])
    out.append(0)
    return out


def strtoint(c):
    '''
    param:
            c:2d list
    return:
            a 2d list
    convert str in the 2d list to int
    '''
    for i in range(len(c)):
        for j in range(len(c[i])):
            c[i][j] = int(c[i][j])

    return c


def parseCNF(file1):
    '''
    param: 
        a cnf file
    return: 
        list: with lenth n as num of clauses
        int: indicating num of variables in the file
    parse the cnf file into a 2d list for further use
    '''
    with open(file1) as f:
        f.readline() #for the first line
        content = f.readlines()
        content = [x.strip() for x in content] 
    # print(content[0])
    clausenum = int(content[0][-3:])
    varnum = int(content[0][-5:-3])
    # print("check:",clausenum)

    #sanity check
    check = content[clausenum].split()[0]
    try: int(check)
    except ValueError: print("ERROR IN PARSING!!")

    content = content[1:clausenum+1]

    #strip 0
    for i in range(clausenum):
        content[i] = content[i][:-2]

    #into 2d
    c2d = []
    for i in range(clausenum):
        c2d.append(content[i].split())

    return c2d, varnum


def readfiles(path):
    '''
    param: 
        str:the path of the folder
    return: 
        a list of all the files in that folder
    read all files end with *cnf within a folder
    '''
    path = path+"/*.cnf"
    cnffiles = []
    for file in glob.glob(path):
        cnffiles.append(file)

    return cnffiles

def DeepCopy(CopyFrom):
    '''
    deep copy
    '''
    return copy.deepcopy(CopyFrom)

def unitclausessearch(c2d, clausenum):
    '''
    param:
        2d list: filled with cnf clauses
        int:len of the 2d list
    return: 
        a list with unit clause var
            empty list if not found
    search for unitclause
    '''
    '''
    #sanity check for unit clause search
    c2d = [['-1'], ['-5', '-7', '9'], ['7', '9', '1'], ['-3', '2', '7'], ['-8', '-2', '5'], ['6', '1', '-9'], ['-4', '5', '3'], ['-1', '4', '-6'], ['5', '2', '7'], ['-10', '-7', '-5'], ['-10', '-3', '8'], ['1', '-8', '-5'], ['5', '6', '-7'], ['8', '-5', '-2'], ['4', '-8', '-2'], ['8', '-2', '9'], ['8', '-2', '3'], ['1', '8', '-9'], ['-1', '2', '4'], ['-7', '-8', '4'], ['-9', '1', '-2'], ['-9', '-3', '-10'], ['9', '10', '7'], ['9', '2', '-1'], ['3', '-10', '1'], ['-4', '9', '-7'], ['-8', '7', '-10'], ['7', '-3', '2'], ['7', '-9', '8'], ['5', '-10', '-1'], ['-9', '-3', '6'], ['5', '-8', '4'], ['-4', '-10', '1'], ['-4', '6', '1'], ['-10', '9', '7'], ['-9', '7', '-6'], ['-1', '-5', '-3'], ['7', '-8', '2'], ['6', '-2', '-1'], ['-9', '4', '-6']]
    '''
    unitclausevar = []
    for i in range(clausenum):
        if(len(c2d[i]) == 1):
            unitclausevar.append(c2d[i][0])
    #check
    if(unitclausevar != []):
        # print("UNIT CLAUSE FOUND...", unitclausevar)
        pass

    return unitclausevar

def puresymbolsearch(c2d, clausenum, varnum):
    '''
    param:
        2d list: filled with cnf clauses
        int: len of the 2d list
        int: num of vars overall
    return: 
        a list with pure symbol var
            empty list if not found
    search for puresymbol
    '''
    #search for puresymbol

    #find the var num in the curr list
    temp_list = functools.reduce(lambda x,y :x+y , c2d)
    temp_list = map(abs, temp_list) #only want the abs val - elimate -
    temp_list = list(set(temp_list))
    varnum = len(temp_list)
    
    # puresymbolvar = [0] * varnum #0 for not pure, 1 for pure
    puresymbolvar = []
    neg_temp = []
    pos_temp = []
    for i in range(clausenum):
        for j in range(len(c2d[i])):
            if(c2d[i][j] < 0):
                neg_temp.append(abs(int(c2d[i][j])))
            else:
                pos_temp.append(int(c2d[i][j]))
    neg_temp = list(set(neg_temp)) #keep unique and sorted
    pos_temp = list(set(pos_temp))

    '''
    #sanity check for pure symbol search
    neg_temp = [1, 2,    4, 5, 6, 7, 8, 9]
    pos_temp = [1,    3, 4, 5, 6,    8, 9, 10]
    '''
    if(len(neg_temp) != varnum or len(pos_temp) != varnum):
        mergedlist = neg_temp + pos_temp

        for i in range(varnum):
            if(mergedlist.count(i+1) == 1):
                puresymbolvar.append(i+1)
        # print("PURE SYMBOL FOUND...", puresymbolvar)
    
    return puresymbolvar


def unitconflict(c):
    '''
    see if there is any conflict in the unit var found
    '''
    # c = [9,3,-3,-9]
    var = list(set(c))
    for i in range(len(var)):
        if var[i] in c and var[i]*(-1) in c:
            return True
    return False

def simplify(c2d, var): 
        '''
        param:
                c2d: 2d int list input
                var: int
        ret:
                a simplified 2d list
        delete every clause contains var
        delete every var in the clause contains ~var
        '''
        currindex = var
        temp_popclause = []
        for i in range(len(c2d)):
            if(currindex in c2d[i]):
                #pop the whole clause -step1
                temp_popclause.append(i)
            elif(-1*currindex in c2d[i]):
                #pop only this literal
                temp_i = c2d[i].index(-1*currindex)
                c2d[i].pop(temp_i)
            else:
                pass
        #pop the whole clause -step2
        for i in range(len(temp_popclause)):
            c2d.pop(temp_popclause[i]-i) #-i since pop 1 then the len will shrink by 1
        
        return c2d


def checkContin(c2d):
    '''
    correct incontinous var assignment
    '''
    temp_list = functools.reduce(lambda x,y :x+y , c2d)
    temp_list = map(abs, temp_list) #only want the abs val - elimate -
    temp_list = list(set(temp_list))
    gap = []
    last = 0
    for i in range(len(temp_list)):
        if temp_list[i] != i+1:
            # gap.append(temp_list[i])
            last = temp_list[i]
            break
    last = last-1
    # print(gap)
    if(last != 0):
            for x in range(len(c2d)):
                for y in range(len(c2d[x])):
                    if c2d[x][y] > last:
                        c2d[x][y] = c2d[x][y] - 1
                    elif c2d[x][y] < -1*last:
                        c2d[x][y] = c2d[x][y] + 1  
    # temp_list = functools.reduce(lambda x,y :x+y , c2d)
    # temp_list = map(abs, temp_list) #only want the abs val - elimate -
    # temp_list = list(set(temp_list))
    # print("afte: ",temp_list)     
    return c2d
