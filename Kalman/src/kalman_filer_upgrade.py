import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
iterations = 30
class Model:
	

	def get_A(self, delta_t):
		return [[1,0, delta_t,0],[0,1, 0, delta_t],[0,0, 1, 0],[0,0,0,1]]

	#H matrix, sensor mallfunction
	def get_H(self, delta_t):
		return [[1,0, 0,0],[0,1, 0, delta_t],[0,0,1,0],[0,0,0,1]]

	def get_A_Tranpose(self, delta_t):
		return [[1,0,0,0],[0,1,0,0],[delta_t,0,1,0],[0,delta_t,0,1]]


	def simulate(self):
		kalman_gain = 0
		
		positionX_error = positionY_error = 20
		velocityX_error = velocityY_error = 5
		positionX_error_measure = positionY_error_measure = 25 
		velocityX_error_measure = velocityY_error_measure = 6
		#accelerationX = accelerationY = 2
		#R = [[position_error_measure *
		#position_error_measure,0],[0,velocity_error_measure *
		#velocity_error_measure]]
		#delta_t = 1
		
		
		#A = [[1,0, delta_t,0],[0,1, 0, delta_t],[0,0, 1, 0],[0,0,0,1]]
		#B = [0.5 * delta_t * delta_t, delta_t]
		#A_Tranpose = [[1,0,0,0],[0,1,0,0],[delta_t,0,1,0],[0,delta_t,0,1]]
		#I = H = H_Tranpose = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]


		model = Model()
		Q = [[positionX_error ** 2,0,0,0],
																[0, positionY_error ** 2,0,0],
																[0,0,velocityX_error ** 2,0],
																[0,0,0,velocityY_error ** 2]]	


		R = [[positionX_error_measure ** 2,0,0,0],
																[0, positionY_error_measure ** 2,0,0],
																[0,0,velocityX_error_measure ** 2,0],
																[0,0,0,velocityY_error_measure ** 2]]

		
		mean = [0,0,0,0]
		state_init = [4000,4000,280,280]
		states_predicted = []
		measures_defect = []
		for x in range(1, iterations + 1):
			#calculate predicted state
			A = model.get_A(x)
			A_Tranpose = model.get_A_Tranpose(x)
			w = np.random.multivariate_normal(mean,Q)
			state_predicted = np.add(np.matmul(A,state_init), w)
			states_predicted.append(state_predicted)
			#calculate predicted measurement
			H = model.get_H(x)
			e = np.random.multivariate_normal(mean,R)
			measure_defect = np.add(np.matmul(H,state_init), e)
			measures_defect.append(measure_defect)
		return states_predicted,measures_defect
#kalman_gain = model.cal_kg(error_covariance_predicted, H, H_Tranpose,
        #R)
        #adjusted_state = model.cal_adjusted_state(state_predicted,
        #kalman_gain, measurements[1])
        #adjusted_error_covariance =
        #model.cal_adjusted_error_covariance(error_covariance_predicted,
        #kalman_gain, I, H)




if __name__ == "__main__":
	m = Model()
	(states_predicted,measures_defect) = m.simulate()
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')	
	ax.set_xlabel('time')
	ax.set_ylabel('x')
	ax.set_zlabel('y')

	time_steps = range(1, iterations + 1)
	ax.scatter(time_steps, [x[0] for x in states_predicted],  [x[1] for x in states_predicted], c = 'b', marker = 'o')
	ax.scatter(time_steps, [x[0] for x in measures_defect],  [x[1] for x in measures_defect], c = 'r', marker = 'o')

	plt.show()