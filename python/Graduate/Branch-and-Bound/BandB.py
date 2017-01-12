import numpy as np
from gurobipy import *

class BranchAndBound:
    def __init__(self, f, A, b, model, Maximize):
        #we will maximize f(x), st Ax <= b
        #for now, we will assume b is element-wise >= 0
        
        m,n = A.shape
        self.A = A
        self.b = b
        self.f = f
        self.Maximize = Maximize
        self.n = n
        self.m = model
        self.current_state = np.zeros(n)
        # Populate our initial list of valid state directions
        self.upper = -np.inf
        I = np.eye(n)
        self.valid_steps = []
        for i in xrange(n):
            self.valid_steps.append(I[i,:])
        self.best_state = np.zeros(self.n)
        self.current_best = f(self.best_state)

    def Branch(self):

        valid_states = [self.current_state]
        
        while len(valid_states) > 0:
            copy = self.valid_steps[:]
            current = valid_states[0]
                
            while len(copy) > 0:
                bad_ind = []
                
                for ind,j in enumerate(copy):
                    current += j
                    
                    # checks feasibility
                    # check if current is our best yet
                    if np.all(self.A.dot(current) <= self.b) & self.Bound(current, False):
                        if self.Maximize:
                            if (self.f(current) > self.current_best):
                                self.best_state = current
                                self.current_best = self.f(current)
                        
                        else:
                            if (self.f(current) < self.current_best):
                                self.best_state = current
                                self.current_best = self.f(current)
                        valid_states.append(current)
                    else:
                        bad_ind.append(ind)
                        current -= j
            # Removed the root from the list of valid
                if len(valid_states) > 0:
                    del valid_states[0]
                else:
                    break;
                bad_ind.reverse()
                for j in bad_ind:
                    del copy[j]

        print valid_states

    def Bound(self, fixed_state,Heur = False):

        try:
            M = self.m.copy()
            # Change gurobi model to LP
            vars_ = M.getVars()
            for i in vars_:
                i.Vtype = 'c'

            for i in xrange(self.n):
                #if fixed_state[i] != 0:
                #    M.addConstr(vars_[i], GRB.EQUAL, fixed_state[i])

                if fixed_state[i] < 0:
                    M.addConstr(vars_[i] <= fixed_state[i])
                if fixed_state[i] > 0:
                    M.addConstr(vars_[i] >= fixed_state[i])
            
            M.update()
            M.optimize()
            
            lin_max_val = M.objVal
            if lin_max_val > self.upper:
                self.upper = lin_max_val               
 
            if self.Maximize:
                if lin_max_val >= self.upper:
                    return True
                else:
                    return False
            else:
                if lin_max_val <= self.upper:
                    return True
                else:
                    return False
 
        except GurobiError:
            print('Error reported')
            return False


    def check_feasilbility(self,M):
        
        try: 
            vars_ = M.getVars()
            for i in vars_:
                i.Vtype = 'i'

            M.update()
            M.optimize()

            M.update()
            M.optimize()
            return True

        except GurobiError:
            print('Error reported')
            return False