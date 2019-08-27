'''
2019/5/4
implement value iteration and policy iteration 
apply these to a simple Markov Decision Process (MDP).
'''
import numpy as np
import copy 

ACTIONS = np.arange(0,4) #top, btm, left, right 
APROB = [0.7,0.2,0.1] #succeed, opposite, stay
GAMMA = 0.95 #DISCOUNT
EPSILON = 1e-4  #error bound
ITER = 2000

def main():
    #init the MDP
    STATES,R,P = initM()
    T = initT()

    #run the val iter
    policy1, U1 = valueiter(STATES,R,P,T)
    #run the policy iter
    policy2, U2 = policyiter(STATES,R,P,T)

    #print the result
    print(R)
    print(STATES)

    print("-------")
    print("val iter...")
    printP(policy1)
    print(U1)
    print()
    print("policy iter...")
    printP(policy2)
    print(U2)
    print("-------")
    
    print("finished...")



def valueiter(STATES,R,P,T): 
    '''
    param: 
        STATES - 2d, Rewards - 1d, P - 1d(not used), T - 3d
    return:
        Policy - 2d, U - 1d
    use value iteration to find optimal policy for MDP
    '''
    # print()
    # print("------val iter--------") 
    U1 = np.zeros(16) #initialize U with zeros
    # U1.fill(1)
    U = copy.deepcopy(U1)
    delta = 0
    #mapp the states and rewards to 1d
    states = sorted(STATES.reshape(16))
    rewards =  np.flip(R)
    rewards = np.asarray([np.flip(i).tolist() for i in R])
    rewards = np.flip(rewards.reshape(16))
    delta = max(0, abs(U1[0] - U[0]))

    #test for states
    # S_c = np.flip(STATES)
    # S_c = np.asarray([np.flip(i).tolist() for i in STATES])
    # S_c = np.flip(S_c.reshape(16))
    #map rewards from 2d to 1d

    #bellman update
    delta = 0
    # print("....compute u...")
    for k in range(ITER):
        U = copy.deepcopy(U1)
        #for all states 0-15
        for s in states:
            actionsum = np.zeros(4)
            for a in ACTIONS:
                #get the arg max of a
                for s1 in states:
                    actionsum[a] += T[s1][s][a] * U[s1]
                    # print(actionsum[a])
                    # print("Transition:",s1,s,a,T[s1][s][a])
                # print(actionsum)
            U1[s] = rewards[s] + GAMMA*max(actionsum) #update the ultility for next state
            delta = max(delta, abs(U1[s] - U[s])) #cal the error
           
        # print(delta)
        #check the converge error 
        if(delta < EPSILON * (1 - GAMMA) / GAMMA):
            print("!!!- reach the delta")
            break

    # print("the U is: ",U)
    # print("....map to pi...")
    #Now obtain optimal policy pi* from utility function U
    policy = np.zeros(16)
    for s in states:
        avals = np.zeros(4)
        #find the optimal action
        for a in ACTIONS:  
            for s1 in states:
                avals[a] += T[s1][s][a]*U[s1]
        # print(s, avals)
        policy[s] = np.argmax(avals) #match the optimal's index

    #map policy
    
    #mapped to 1d to display
    policy = map1dto2d(policy)
    # print(policy)
    # printP(policy)

    return policy, U


def policyiter(STATES, R, P, T):
    '''
    param: 
        STATES - 2d, Rewards - 1d, P - 1d(not used), T - 3d
    return:
        Policy - 2d, U - 1d
    use policy iteration to find optimal policy for MDP
    '''
    # print()
    # print("------pol iter--------") 
    #reshape the input...
    states = sorted(STATES.reshape(16))
    rewards =  np.flip(R)
    rewards = np.asarray([np.flip(i).tolist() for i in R])
    rewards = np.flip(rewards.reshape(16))
    
    #init Uiltility and policy
    U = np.zeros(16)
    policy = np.zeros(16)
    policy.fill(2) #randomly pick a val
    policy = policy.astype(int)
    # print(policy)
    count = 0
    
    while True:
        # count += 1
        # if(count > 20):
        #     break
        #eval curr policy
        for k in range(ITER):
            for s in states:
                total = 0
                #find the curr optimal
                for s1 in states:
                    a = policy[s]
                    # print(a)
                    # print(T[s1][s][a])
                    total += T[s1][s][a]  * U[s1]
                U[s] = rewards[s] + GAMMA*total #cal the utility based on A

        unchanged = True #until converge
        for s in states:
            actionsum = np.zeros(4) 
            #find all curr A
            for a in ACTIONS:
                for s1 in states:
                    actionsum[a] += T[s1][s][a] * U[s1]
            maxA = max(actionsum)
            bestA = max(np.where(actionsum == maxA)[0])
            # print(bestA)
            #keep the optimal A
            if maxA > actionsum[policy[s]]:
                policy[s] = bestA
                unchanged = False
            # print(actionsum)

        if unchanged:
            # print("not changing")
            break

    #mapped to 2d to display
    policy = map1dto2d(policy)

    return policy, U

def printP(l1):
    '''
    param:
        l1 - 2d
    print the action in symbols and format in 2d 
    '''
    print("---------------")
    out = []
    for i in range(len(l1)):
        temp = []
        for j in range(len(l1[i])):
            if l1[i][j] == 0:
                temp.append("^")
            elif l1[i][j] == 1:
                temp.append("v")
            elif l1[i][j] == 2:
                temp.append("<")
            else:
                temp.append(">")
        out.append(temp)
    
    print('\n'.join([''.join(['{:4}'.format(item) for item in row]) 
      for row in out]))
    print("---------------")

def map1dto2d(l1):
    '''
    param:
        l1 - 1d
    return:
        2d list of actions
    '''
    l1 = np.asarray(l1)
    M = l1.reshape(4, 4)
    M = np.flip(M)
    M = np.asarray([np.flip(i).tolist() for i in M])
    #flip in order to mapped to the format correspondign to set up to display
    return M


def initT():
    '''
    initialize a transition matrix
    a three dim array 16*16*4
    T[nextState][currState][Action]
    '''
    T = np.zeros(16*16*4).reshape(16, 16,4)
    # diag1 = np.zeros(16*4).reshape(16,4)
    # diag1.fill(0.1)
    # # print(diag1)

    row,col= np.diag_indices(T.shape[0])
    # col = np.diag_indices(T.shape[1])
    T[row, col] =  np.array([0.1,0.1,0.1,0.1])

    #right/left wall
    wallcells = np.asarray([0,4,8,12, 3,7,11,15])
    wallcells_n = np.asarray([1,5,9,13, 2,6,10,14])
    T[wallcells_n, wallcells] =  np.array([0,0,0.9,0.9]) #go right/left
    #top/btm wall
    wallcells = np.asarray([0,1,2,3 ,12,13,14,15])
    wallcells_n = np.asarray([4,5,6,7, 8,9,10,11])
    T[wallcells_n, wallcells] =  np.array([0.9,0.9,0,0]) #go up/down
    #middle
    cells = np.asarray([5,6,9,10])
    cells_n = np.asarray([9,10,13,14])
    T[cells_n, cells] =  np.array([0.7,0.2,0,0]) #go up
    cells_n = np.asarray([1,2,5,6])
    T[cells_n, cells] =  np.array([0.2,0.7,0,0]) #go down
    cells_n = np.asarray([4,5,8,9])
    T[cells_n, cells] =  np.array([0,0,0.7,0.2]) #go left
    cells_n = np.asarray([6,7,10,11])
    T[cells_n, cells] =  np.array([0,0,0.2,0.7]) #go right

    #for middle
    midcells = np.asarray([8, 4, 11, 7])
    midcells_n = np.asarray([12, 8, 15, 11])
    T[midcells_n, midcells] =  np.array([0.7,0.2,0,0])
    midcells = np.asarray([8, 11])
    midcells_n = np.asarray([4, 7])
    T[midcells_n, midcells] =  np.array([0.2,0.7,0,0])

    midcells = np.asarray([4, 7])
    midcells_n = np.asarray([0, 3])
    T[midcells_n, midcells] =  np.array([0.2,0.7,0,0])
    
    midcells = np.asarray([13, 14, 1, 2])
    midcells_n = np.asarray([12, 13, 0, 1])
    T[midcells_n, midcells] =  np.array([0,0,0.7,0.2])

    midcells = np.asarray([13, 14, 1, 2])
    midcells_n = np.asarray([14, 15, 2, 3])
    T[midcells_n, midcells] =  np.array([0,0,0.2,0.7])

    # print()
    # print("-----")
    # for i in range(16):
    #     print(i , T[15][i])
    # print("-----")
    # print()
    # # print(T[10][11])

    return T


def initM():
    '''
    init the MDP
    return M, Reward, Prob(not used)
    '''
    #GET STATE MATRIX
    M = np.arange(0,16).reshape(4, 4)
    M = np.flip(M)
    M = np.asarray([np.flip(i).tolist() for i in M])
    R = copy.deepcopy(M)

    #PROB ACTION MATRIX 3D each cell in M corresponding to [.25 .25 .25 .25]four direcion
    P = np.zeros(4*16).reshape(4,4,4)
    P.fill(1/4)

    #GET CORRESPONDING REWARDS 
    mask1 = np.arange(1,16,2)
    mask2 = np.arange(0,15,2)
    mask2 = np.delete(mask2,5) #delete 10
    mask2 = np.delete(mask2,5) #delete 12
    for i in mask1:
        R[R == i] = 50
    for i in mask2:
        R[R == i] = 0
    R[R == 10] = 100
    R[R == 12] = 200
    
    return M,R,P

main()
