import numpy as np
import matplotlib.pyplot as plt



class Model:
	def cal_error_covariance_predicted(self, A, P, A_Tranpose):
			error_covariance = np.matmul(np.matmul(A, P), A_Tranpose)
			#do not know the correct way removing covariance, replace with 0 row by row
			#row1
			error_covariance[0][1] = 0
			error_covariance[0][2] = 0
			error_covariance[0][3] = 0
			#row2
			error_covariance[1][0] = 0
			error_covariance[1][2] = 0
			error_covariance[1][3] = 0
			#row3
			error_covariance[2][0] = 0
			error_covariance[2][1] = 0
			error_covariance[2][3] = 0
			#row4
			error_covariance[3][0] = 0
			error_covariance[3][1] = 0
			error_covariance[3][2] = 0
			return error_covariance

	def run(self):
		kalman_gain = 0
		measurements = [[4000,4000,280,280],[4260,4260,282,282],[4550,4550,285,285]]
		positionX_error = positionY_error = 20
		velocityX_error = velocityY_error = 5
		positionX_error_measure = positionY_error_measure = 25 
		velocityX_error_measure = velocityY_error_measure = 6
		accelerationX = accelerationY = 2
		#R = [[position_error_measure *
		#position_error_measure,0],[0,velocity_error_measure *
		#velocity_error_measure]]
		delta_t = 1
		
		A = [[1,0, delta_t,0],[0,1, 0, delta_t],[0,0, 1, 0],[0,0,0,1]]
		B = [0.5 * delta_t * delta_t, delta_t]
		A_Tranpose = [[1,0,0,0],[0,1,0,0],[delta_t,0,1,0],[0,delta_t,0,1]]
		I = H = H_Tranpose = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
		initial_error_covariance = [[positionX_error ** 2,0,0,0],
																[0, positionY_error ** 2,0,0],
																[0,0,velocityX_error ** 2,0],
																[0,0,0,velocityY_error ** 2]]
			
			
		state_predicted = np.matmul(A,measurements[0])
		model = Model()
		error_covariance_predicted = model.cal_error_covariance_predicted(A,initial_error_covariance, A_Tranpose)
		print(error_covariance_predicted)  
#kalman_gain = model.cal_kg(error_covariance_predicted, H, H_Tranpose,
        #R)
        #adjusted_state = model.cal_adjusted_state(state_predicted,
        #kalman_gain, measurements[1])
        #adjusted_error_covariance =
        #model.cal_adjusted_error_covariance(error_covariance_predicted,
        #kalman_gain, I, H)




if __name__ == "__main__":
	m = Model()
	m.run()
