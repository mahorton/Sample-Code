import numpy as np
import KalmanFilter
from matplotlib import pyplot as plt


def main():

	""" 
	In this example, I demonstrate the functions of my Kalman Filter class as an application of 
	tracking projectile motion using noisy observations.
	"""
	# initialize the projectile system parameters and kalman filter
	F = np.array([[1.,0,.1,0.],[0,1.,0.,.1],[0.,0.,1.,0.],[0.,0.,0.,1.]])
	H = np.array([[1.,0.,0.,0.],[0.,1.,0.,0.]])
	u = np.array([0.,0.,0.,-.98])
	Q = np.identity(4) * .1
	R = np.identity(2) * 5000.
	kal = KalmanFilter.KalmanFilter(F,Q,H,R,u)

	# simulate state sequence and observations
	x0 = np.array([0.,0.,300.,600.])
	N = 1250
	states, obs = kal.evolve(x0,N)
	plt.plot(states[0,:],states[1,:],'b')

	# estimate states given observations 200 through 800
	x0 = np.array([obs[0,200],obs[1,200],np.sum(np.diff(obs[0,200:208]))/.8,np.sum(np.diff(obs[1,200:208]))/.8])
	P0 = 10e6*Q
	z = obs[:,200:800]
	out, norms = kal.estimate(x0,P0,z,True)
	plt.plot(out[0,200:800],out[1,200:800],'g')
	plt.scatter(obs[0, 200:800],obs[1, 200:800],c='r')

	# predict states until impact given state estimate at time 800
	x0 = out[:,-1]
	predict = kal.predict(x0,450)
	plt.plot(predict[0, :],predict[1,:],'y')

	# predict states from origin given state estimate at time 250
	x0 = out[:,50]
	rewind = kal.rewind(x0,250)
	plt.plot(rewind[0,:],rewind[1,:],'c')

	plt.show()

if __name__ == "__main__":
	main()