'''
4/12/2019
For the Bayesian network
computes and prints out the probability of any combination of events
given any other combination of events.


(E) --->     ---> (J)
(B) ---> (A) ---> (M)
!!!!!!!!!!!!!! 0 - T, 1 - F !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
'''
#!/usr/bin/env python
import sys  # https://www.pythonforbeginners.com/system/python-sys-argv
import numpy as np
import random
import itertools
ITER = 100000 #num of iteration for reject sampling

def main():
    bn = initBN()
    pre, cond = parseargs()
    pre = sorted(pre)
    cond = sorted(cond)
    out_b = [1 if i[-1:] == "f" else 0 for i in pre]
    out_l = [i[:1] for i in pre]

    varlist = sorted(['A', 'B', 'E', 'J', 'M'])
    # print(varlist)
    probs,flags = fulljoint(varlist,bn)

    # diff = list(set(varlist) - set(out_l))
    # print(diff)
    print(out_b)
    print(out_l)
    # partial = genP(varlist,out_l,out_b, len(diff)).tolist()
    # print(partial)
    #if no condition involved

    # if(len(cond) == 0):
    #     r = getJoint(varlist, out_l, out_b, probs, flags)
    #     print("#result: ", r)
    #     return r
    # else:
    #     joint = sorted(pre + cond)
    #     out_b = [1 if i[-1:] == "t" else 0 for i in joint]
    #     out_l = [i[:1] for i in joint]
    #     # print(out_b)
    #     # print(out_l)
    #     r = getJoint(varlist, out_l, out_b, probs, flags)
    #     out_b_1 = [1 if i[-1:] == "t" else 0 for i in cond]
    #     out_l_1 = [i[:1] for i in cond]
    #     r1 = getJoint(varlist, out_l_1, out_b_1, probs, flags)
    #     result = r/r1
    #     print("#result: ", result)
    #     return result
    result = exact(varlist, pre, cond,out_l, out_b, probs, flags)
    # reject_joint(bn, out_l,out_b)
    print("#result: ", result)
    print("finished...")
    print()
    return result


def exact(varlist, pre, cond,out_l, out_b, probs, flags):
    '''
    the process of exact inference
    '''
    if(len(cond) == 0):
        r = getJoint(varlist, out_l, out_b, probs, flags)
        # print("#result: ", r)
        return r
    else:
        joint = sorted(pre + cond)
        out_b = [1 if i[-1:] == "t" else 0 for i in joint]
        out_l = [i[:1] for i in joint]
        # print(out_b)
        # print(out_l)
        r = getJoint(varlist, out_l, out_b, probs, flags)
        out_b_1 = [1 if i[-1:] == "t" else 0 for i in cond]
        out_l_1 = [i[:1] for i in cond]
        r1 = getJoint(varlist, out_l_1, out_b_1, probs, flags)
        result = r/r1
        # print("#result: ", result)
        return result

def getJoint(varlist, out_l, out_b, probs, flags):
    '''
    matched the command line input to the full joint dist and gives out results
    '''
    r = 0
    diff = list(set(varlist) - set(out_l))
    partial = genP(varlist,out_l,out_b, len(diff)).tolist()
    for clause in partial:
            found = flags.index(clause)
            # print("!",found)
            raw = probs[found]
            result = 1
            for i in raw:
                result *= i
            r += result

    return r  


def genP(varlist,l,b,freenunm):
    '''
    based on the input, filled the gaps if the input param is less than 5
    '''
    # print(varlist)
    # print(l, b)
    if freenunm == 0:
        return np.asarray([b])
    else:
        full = np.asarray([list(i) for i in itertools.product([0, 1], repeat=freenunm)])
        # print(full)
        indexlist = [varlist.index(l[i]) for i in range(len(l))]
        # print(indexlist)
        for i in range(len(l)):
            full = np.insert(full, indexlist[i], b[i], axis = 1)
        # print(full)
        return full

def computejointprob(l, bn):
    '''
    compute the joint prob distribution for a single condition
    '''
    #P(B=T)P(E=T)P(A=T|B=T,E=T)P(J =T| A=T)P(M =F| A=T)
    prob = []
    fil = [x for _,x in l] #boolean val for abe
    varlist = [x for x,_ in l] #boolean val for abe
    # print(varlist)
    # print(fil)
    for i in range(len(bn)):
        if bn[i].parent == []:
            prob.append(bn[i].prob[fil[i]])
        else:
            temp = []
            for n in bn[i].parent:
                temp.append(varlist.index(n.name))
            temp_flag = [fil[x] for x in temp] #grab the corresponding binary flag
            if len(temp_flag) > 1: #abbrv..........otz
                p = bn[i].prob[temp_flag[0]][temp_flag[1]]
                prob.append(p)
            else:
                prob.append(bn[i].prob[0][temp_flag[0]])
        
    return prob, fil


def fulljoint(varlist,bn):
    '''
    compute the full joint distribution with 2**5 dim
    '''
    # full = np.zeros((2**5,5))
    full = np.asarray([list(i) for i in itertools.product([0, 1], repeat=5)])
    print(full.shape)
    full_comb = []
    iternum = full.shape[0]
    flags = []
    probs = []
    for i in range(iternum):
        mapped = zip(varlist,full[i])
        mapped = sorted(set(mapped))
        prob, fil = computejointprob(mapped,bn)
        probs.append(prob)
        flags.append(fil)
        full_comb.append(mapped)
    # print(np.asarray(probs))
    return  probs,flags

def reject_joint(bn, events, flags):
    '''
    use reject sampling to approx prob
    '''
    print("#events:",events)
    print("#flags:",flags)
    diff = []
    if(len(events) != 5):
        varlist = sorted(['A', 'B', 'E', 'J', 'M'])
        diff = list(set(varlist) - set(events))
        print(diff)
        for i in diff:
            index = varlist.index(i)
            flags.insert(index, 0)
        events = varlist
        

    result = 0
    for iter in range(ITER):
        count = 0
        gen = [np.random.uniform(0, 1) for i in range(len(events))]
        # prrint(gen)
        if(diff != []):
            for i in diff:
                index = varlist.index(i)
                gen.insert(index, 0)
        # print(gen)
        # print(flags)

        for i in range(len(bn)):
            if bn[i].parent == []:
                if events[i] == bn[i].name:
                    # print("name", bn[i].name, flags[i])
                    temp_sample = gen.pop()
                    # print(bn[i].prob[flags[i]], temp_sample)
                    if temp_sample < bn[i].prob[flags[i]]:
                        count += 1
                
            else:
                temp = []
                for n in bn[i].parent:
                    temp.append(flags[events.index(n.name)])
                if len(temp) > 1: #abbrv..........otz
                    p = bn[i].prob[temp[0]][temp[1]]
                    temp_sample = gen.pop()
                    # print(temp_sample, p)
                    if(temp_sample < p):
                        count += 1
                else:
                    p = bn[i].prob[0][temp[0]]
                    temp_sample = gen.pop()
                    # print(temp_sample, p)
                    if(temp_sample < p):
                        count += 1
            # print("##", count)          
            if(count == 5):
                # print("##", count)  
                result += 1

    print("!!!!!!!!!",result/ITER)

class BNNode:
    '''
    the node structure in the bayes net with name, parents and prob associated
    '''
    def __init__(self, name, parent, prob):
        self.name = name #str
        self.parent = parent #list
        self.prob = prob
    def assignProb(self, prob):
        self.prob = prob

def initBN():
    '''
    init a Bayes net
    '''
    e = BNNode('E',[],[0.03, 0.97])
    b = BNNode('B',[],[0.02, 0.98])

    a = BNNode('A',[b,e], 
        np.array([
        [0.97, 0.92], # a | b = t, e = t
        [0.36, 0.03]]))

    j = BNNode('J',[a], 
        np.array([
        [0.85, 0.07]]))# j | a = f

    m = BNNode('M',[a], 
        np.array([
        [0.69, 0.02] ]))# j | a = f
    bn = [a,b,e,j,m]
    return bn
         
class BN:
    '''
    not used
    '''
    p_e = np.array([0.03, 0.97]) #index 0-T, 1-F
    p_b = np.array([0.02, 0.98])
    p_j_given_a = np.array([
        [0.85, 0.15], # j | a = t
        [0.07, 0.93] ])# j | a = f
    p_m_given_a = np.array([
        [0.69, 0.31], # j | a = t
        [0.02, 0.98] ])# j | a = f
    p_a_given_b_e = np.array([
        [0.97, 0.03], # a | b = t, e = t
        [0.92, 0.08], # a | b = t, e = f
        [0.36, 0.64], # a | b = f, e = t
        [0.03, 0.97] ])# a | b = f, e = f

def parseargs():
    '''
    handle the command line input to lists
    '''
    args = ['bn.py', 'Jt', 'Af', 'given', 'Bt', 'Ef'] # default - test for cond
    # args = ['bn.py', 'Bt', 'At', 'Mt', 'Jt', 'Et']  # test for joint
    # args = ['bn.py', 'Af', 'Et']
    if(len(sys.argv) > 1 and len(sys.argv) < 7): 
        args = sys.argv

    args = (args)[1:]
    pre = args
    cond = []
    if('given' in args):  # conditional
        given = args.index('given')
        pre = args[:given]
        cond = args[given+1:]
    return pre, cond

main()
