import numpy as np

#error function to minimise
def error_function(X,Y,Z):
	err = 0.5*np.linalg.norm(J*(U-np.dot(X,Y.transpose())),ord='fro')**2 + (0.5*alpha)*np.linalg.norm(T-np.dot(X,Z.transpose()),ord='fro')**2+(0.5*beta)*(np.linalg.norm(X,ord='fro')**2+np.linalg.norm(Y,ord='fro')**2+np.linalg.norm(Z,ord='fro')**2)
	return err

def gradient_descent(epsilon,max_iter):
	#initialize X,Y,Z with random number in range (0,1)
	
	while error_function
