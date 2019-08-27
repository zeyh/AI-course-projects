from fcn import *
import numpy as np

def walksatprocess(c, p, iterationnum):
    '''
    param:
        2d list cnf
        int - max iteration
    return:
        sat or not sat
        assigned val or []
    tell if sat or not sat
    '''
    currval = 0
    count = 0
    for i in range(iterationnum):     
        currval = evaluate(c,p)
        count += 1
        if(currval == len(c)):
            # print(currval)
            return "SAT", p, count
        else:
            ran = random.randint(0, 1) 
            if(ran):
                #get maxmize p
                p, currval  = genBest(c,p)
            else:
                #flip randomly
                ranindex = random.randint(0, 9)
                p[ranindex] = -1*p[ranindex]

    # print(currval)
    return "UNSAT",[],count

COUNT_H = []
def hill(c,p,n):
        '''
        param:
            2d list cnf
            1d assign p 
        return:
            best p 
            best heuristic val
        twick each element in p to get best val
        '''
 
        COUNT_H.append(1)
        currval = evaluate(c,p)
        # print(currval)
        # print(currval)
        #generate next best assign
        nextp, nextval  = genBest(c,p)
        if(nextval > currval):
            return hill(c,nextp,n+1)
        else:
            return p,currval,n


def hillprocess(c, iterationnum):
    '''
    param:
        2d list cnf
        int - max iteration
    return:
        sat or not sat
        assigned val or []
    tell if sat or not sat
    '''
    maxval = -1
    count = 0
    for i in range(iterationnum):
        count+= 1
        p = ranGen(c)

        p, currval, n = hill(c,p,0)
        # print(currval)
        if(currval > maxval):
            maxval = currval
        if(maxval == len(c)):
            # print(p)
            # print(i)
            # print(n)
            return "SAT", p, currval
        else:
            return "UNSAT", [], currval





TREE = []
SOL = []
COUNT = []
def DPLL(c2d, varnum, varlist):
    '''
    param: 
           list: 2d lists
           int: indicate num of vars
    return: 
           str: like "SAT -1 -2 -3 -4 -5 6 7 -8 -9 -10 0"
                "UNSAT" if cannot find answer
    use DPLL to search for answer for the input list
    '''
    #copy the state visited to the tree
    c2d_c = DeepCopy(c2d)
    varlist_c = DeepCopy(varlist)
    TREE.append(c2d_c)
    SOL.append(varlist_c)

    #rmv empty list in 2d list for bug like: [[],[],[123]] 
    c2d = [x for x in c2d if x != []] 
    #check--------------------------------------------
    if(len(c2d) == 0 or ((-1 not in varlist))):
        #if it is empty then satisfied
        # print("SAT1",varlist)
        return ("SAT", varlist)

    else:
        COUNT.append(1) 
        #detect if there is pure symbol or unit clause
        puresymbolvar = puresymbolsearch(c2d, len(c2d), varnum)
        unitclausevar = unitclausessearch(c2d, len(c2d))
        #concat unit and pure var into one list
        pureorunit = unitclausevar + puresymbolvar
        pureorunit = list(set(pureorunit))

        #while there exist pure symbol or unit clause
        while(len(pureorunit) != 0):
            #if there is conflict in unit clause
            if(unitconflict(unitclausevar)):
                # print("go back")
                if len(TREE) == 0:
                    return "UNSAT"
                else:
                    for temp in range(len(pureorunit)-1):
                        if len(TREE) == 0:
                            return "UNSAT"
                        else:
                            TREE.pop()
                            SOL.pop()
                    if len(TREE) == 0:
                        return "UNSAT"
                    else:
                        c2d_next = TREE.pop()
                        varlist_next = SOL.pop()
                        init_p = c2d[0][0] #arbitrary choose 1 var to flip and try
                        return DPLL(simplify(c2d_next,-1*init_p), varnum, varlist_next)
            else:
                for i in range(len(pureorunit)): 
                    #check again----------------------------------------------  
                    if(len(c2d) == 0 or (-1 not in varlist)):
                        # print("SAT2",varlist)
                        return ("SAT", varlist)
                    #simplify the clause based on the pure or unit list
                    popnext = pureorunit.pop()
                    c2d = simplify(c2d, popnext)
                    #update the varlist accordingly : 0 neg, 1 pos
                    # print(c2d)
                    if(abs(popnext) != 0):
                        varlist[abs(popnext)-1] = int((popnext / abs(popnext) + 1) / 2)
                    else:
                        varlist[0] = 1
    
        #check again------------------------  
        # #rmv empty list in 2d list for bug like: [[],[],[123]] 
        c2d = [x for x in c2d if x != []]   
        if(len(c2d) == 0 or (-1 not in varlist)):
            # print("SAT3",varlist)
            return ("SAT", varlist)      

        else:
            #pick the first one as the literal being set
            # print(c2d)
            init_p = c2d[0][0]
            #update the varlist
            if(init_p != 0):
                varlist[abs(init_p)-1] = int((init_p / abs(init_p) + 1) / 2)
            else:
                varlist[0] = 1
            if (DPLL(simplify(c2d,init_p), varnum, varlist)[0] == "SAT"): 
                return ("SAT", varlist)
            else:
                return DPLL(simplify(c2d,-1*init_p), varnum, varlist)





def main():
    cnffiles = readfiles(PATH)
    walk = []
    climb = []
    dp = []
    for i in range(32):
        # i = 6
        print(i)
        # i = 29
        # print(i)
        c2d, varnum = parseCNF(cnffiles[i])
        # print(var)
        c2d = strtoint(c2d) 
        c2d = checkContin(c2d)

        p = ranGen(c2d)

        # # tempcount = 0
        # # for j in range(10):
        f,p,count = walksatprocess(c2d, p, 20)
        # # print("walksat: ",f,changeformat(p))
        # # print("#walksat: #C ",count )
        # walk.append(count)

        # prev = len(COUNT_H)
        f1,p1,count1 = hillprocess(c2d, 20)
        # print(len(COUNT_H) )
        # print(count1)
        # climb.append(count1)
        

  
        '''
        # print("hillclimbing: ",f1,changeformat(p1))
        print("#hill: #C ",count1 )


        print("---")
        #initial a var lists of -1, index corresponds to var
        var = [-1] * varnum
        flag = DPLL(c2d,varnum, var)
        if(flag[0] == "UNSAT"):
            print(flag)
        print("#DPLL: #C: ", len(COUNT))
        # print(flag)

        print("finished...")
        print()
        print()
        '''
        
        var = [-1] * varnum   
        # prev = len(COUNT)
        DPLL(c2d,varnum, var)
        # curr = len(COUNT) - prev
        # print(curr)
        # dp.append(len(COUNT))

    # np.savetxt('walksat.csv', walk, delimiter=',', fmt='%d')
    # np.savetxt('climb.csv', climb, delimiter=',', fmt='%d')
    # np.savetxt('dpll.csv', dp, delimiter=',', fmt='%d')
main()


