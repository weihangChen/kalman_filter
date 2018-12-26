import numpy as np
import matplotlib.pyplot as plt
iterations = 50
positionX_error = positionY_error = 20
velocityX_error = velocityY_error = 5
positionX_error_measure = positionY_error_measure = 400
velocityX_error_measure = velocityY_error_measure = 7

Q = [[positionX_error ** 2,0,0,0],
																[0, positionY_error ** 2,0,0],
																[0,0,velocityX_error ** 2,0],
																[0,0,0,velocityY_error ** 2]]	


R = [[positionX_error_measure ** 2,0,0,0],
														[0, positionY_error_measure ** 2,0,0],
														[0,0,velocityX_error_measure ** 2,0],
														[0,0,0,velocityY_error_measure ** 2]]

mean = [0,0,0,0]
state_init = np.array([4000,4000,280,280])

def get_A(delta_t):
	return np.array([[1,0,delta_t,0],[0,1,0,delta_t],[0,0,1,0],[0,0,0,1]])

#H matrix, sensor mallfunction
def get_H(delta_t):
	return np.array([[1,0,delta_t,0],[0,1,0,delta_t],[0,0,1,0],[0,0,0,1]])

def get_H_Error(delta_t):
	return np.array([[1,0,delta_t,0],[0,1,0,delta_t],[0,0,1,0],[0,0,0,1]])

class Model:
	def cal_state_predicted(self, A, state_predicted_current):
			state_predicted_next = np.matmul(A, state_predicted_current)
			return state_predicted_next

	#no convariance, only variance
	def cal_initial_error_covariance(self, error_x, error_vx):
			matrix = [[error_x * error_x, 0],[0, error_vx * error_vx]]
			return matrix

	def cal_error_covariance_predicted(self, A, P, A_Tranpose):
			error_covariance = np.matmul(np.matmul(A, P), A_Tranpose)
			#remove covariance
			error_covariance = np.fliplr(error_covariance)
			np.fill_diagonal(error_covariance,0)
			error_covariance = np.fliplr(error_covariance)
			
			#error_covariance[0][1] = 0
			#error_covariance[0][2] = 0
			#error_covariance[0][3] = 0

			#error_covariance[1][0] = 0
			#error_covariance[1][2] = 0
			#error_covariance[1][3] = 0

			#error_covariance[2][0] = 0
			#error_covariance[2][1] = 0
			#error_covariance[2][3] = 0

			#error_covariance[3][0] = 0
			#error_covariance[3][1] = 0
			#error_covariance[3][2] = 0

			return error_covariance


	def cal_kg(self, error_covariance, H, H_Tranpose, R):
			A = np.matmul(error_covariance, H_Tranpose)
			B = np.add(np.matmul(np.matmul(H, error_covariance), H_Tranpose), R)
			kg = np.divide(A, B)
			kg = np.round(np.nan_to_num(kg),3)
			return kg

	def cal_adjusted_state(self, state_estimate, kalman_gain, state_measured):
			tmp1 = np.subtract(state_measured, state_estimate)
			tmp2 = np.round(np.matmul(kalman_gain, tmp1),1)
			adjusted_state = np.add(state_estimate, tmp2)
			return adjusted_state

	def cal_adjusted_error_covariance(self, convariance_matrix_predicted, K, I, H):
			KH = np.matmul(K, H)
			adjusted_error_covariance = np.multiply(np.subtract(I, KH),convariance_matrix_predicted)
			return np.round(adjusted_error_covariance,1)


	def simulate(self):
		states_predicted = []
		measures_defect = []
		for x in range(1, iterations + 1):
			#calculate predicted state
			A = get_A(x)
			w = np.random.multivariate_normal(mean,Q)
			state_predicted = np.add(np.matmul(A,state_init), w)
			states_predicted.append(state_predicted)
			#calculate predicted measurement
			H = get_H_Error(x)
			e = np.random.multivariate_normal(mean,R)
			measure_defect = np.add(np.matmul(H,state_init), e)
			measures_defect.append(measure_defect)

		return states_predicted,measures_defect





if __name__ == "__main__":
	model = Model()
	(states,measurements) = model.simulate()
	#
	A = get_A(1)
	H = get_H(1)
	states_adjusted = []
	
	state_base = []
	error_covariance_base = []
	I = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

	for k in range(1, iterations):
		if len(state_base) == 0:
			state_base = measurements[0]
		else:
			state_base = state_adjusted
		state_predicted = model.cal_state_predicted(A, state_base)

		if len(error_covariance_base) == 0:
			error_covariance_base = Q
		else:
			error_covariance_base = error_covariance_adjusted
		error_covariance_predicted = model.cal_error_covariance_predicted(A, error_covariance_base, A.transpose())

		kalman_gain = model.cal_kg(error_covariance_predicted, H, H.transpose(), R)
		state_adjusted = model.cal_adjusted_state(state_predicted, kalman_gain, measurements[k])
		error_covariance_adjusted = model.cal_adjusted_error_covariance(error_covariance_predicted, kalman_gain, I, H)
		states_adjusted.append(state_adjusted)
		


	plt.plot([x[0] for x in states], [x[1] for x in states], color = 'blue', marker = 'o')
	plt.plot([x[0] for x in measurements], [x[1] for x in measurements], color = 'red', marker = 'o')
	plt.plot([x[0] for x in states_adjusted], [x[1] for x in states_adjusted], color = 'green', marker = 'o')
	plt.show()