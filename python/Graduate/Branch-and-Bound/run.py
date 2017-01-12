import numpy as np
import BandB
from gurobi import *

"""
Performs integer programming via branch and bound.
Gurobi is used to handle linear program relaxations only.
"""

# change value to change max to min
Maximize = True
if Maximize:
	z = 1
else:
	z = -1

def getData():

	# Choose .mps file to be used:
	model = read("iis-100-0-cov.mps")
	n = model.NumVars
	m = model.NumConstrs
	vars_ = model.getVars()
	constrs = model.getConstrs()
	obj = model.getObjective()
	expr = LinExpr()


	print "n: ", n
	print "obj size: ", obj.size()

	for i in xrange(n):
		expr += vars_[i] * obj.getCoeff(i)
	model.setObjective(expr, GRB.MAXIMIZE)
	obj = model.getObjective()


	def f(x):
		sum_ = 0
		for i in xrange(obj.size()):
			sum_ += x[i] * obj.getCoeff(i)
		sum_ += obj.getConstant()
		return sum_

	# Construct A and b, checking for constraint sense
	A = []
	b = []
	for i in xrange(m):
		row = []
		if constrs[i].Sense == "<": 
			for j in xrange(n):
				row.append(model.getCoeff(constrs[i], vars_[j]))
			A.append(row)
			b.append(np.abs(constrs[i].RHS))
		if constrs[i].Sense == "=":
			row2 = [] 
			for j in xrange(n):
				row.append(model.getCoeff(constrs[i], vars_[j]))
				row2.append(-1*model.getCoeff(constrs[i], vars_[j]))
			A.append(row)
			A.append(row2)
			b.append(np.abs(constrs[i].RHS))
			b.append(-1*np.abs(constrs[i].RHS))
		if constrs[i].Sense == ">": 
			for j in xrange(n):
				row.append(-1*model.getCoeff(constrs[i], vars_[j]))
			A.append(row)
			b.append(np.abs(constrs[i].RHS))

	A = np.array(A)
	b = np.array(b)
	return f, A, b, model

def testProb():
	
	A = 1.*np.array([[1,5,-7,0],[0,0,3,2],[0,2,0,-5],[0,0,0,3],[-1,-2,10,3],[1,0,0,0]])
	b = 1.*np.array([100,100,100,100,100,100])
	m,n = A.shape
	def f(x):
		c = np.array([1,2,4,8])
		return c.dot(x)
	 
	# Create a new model
	m = Model()

	# Create variables
	variables = []
	for i in xrange(n):
		variables.append(m.addVar(vtype=GRB.CONTINUOUS, name="x" + `i`))
	m.update()
	vars_ = m.getVars()

	        
	# Add constraints
	for i in xrange(A.shape[0]):
		expr = LinExpr()
		for j in xrange(n):
			if A[i][j] != 0:
				expr += A[i][j] * vars_[j]
		m.addConstr(expr, GRB.LESS_EQUAL, b[i])
	
	# Set objective            
	expr = LinExpr()
	e = np.eye(n)
	for i in xrange(n):
		expr += f(e[i])*vars_[i]
	if Maximize:
		m.setObjective(expr, GRB.MAXIMIZE)
	else:
		m.setObjective(expr, GRB.MINIMIZE)

	m.update()

	return f, A, b, m


# Switch between test problem and getData for data

#f, A, b, model = getData()
f, A, b, model = testProb()


n = A.shape[1]

def gurobiSol(M):

	# Gurobi solver to check solutions
	vars_ = M.getVars()
	for i in vars_:
		i.Vtype = 'i'

	M.update()
	M.optimize()

	M.update()
	M.optimize()
	vars_ = M.getVars()
	sol = np.zeros(A.shape[1])
	for i in xrange(A.shape[1]):
		sol[i] = vars_[i].X
	return M.objVal, sol

gerSol, sol = gurobiSol(model)


# Create bnb object and run optimization
bnb = BandB.BranchAndBound(f, A, b, model, Maximize)
if bnb.check_feasilbility(model):
	bnb.Branch()
	print "Our solution: ",bnb.best_state
	print "our sol evaluates to: ",f(bnb.best_state)

	print "Gurobi's solution: ", sol
	print "which evaluates to: ", gerSol

else:
	print "problem not feasible for integer program"
