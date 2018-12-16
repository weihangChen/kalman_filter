import numpy as np
import matplotlib.pyplot as plt



class Model:
    #ideal situation where math formula is used to predict the future
    #take accelation into account
    def cal_state_predicted(self, A, state_predicted_current, B, acceleration):
        state_predicted_next = np.matmul(A, state_predicted_current)
        state_predicted_next = np.add(state_predicted_next, np.multiply(B,acceleration))
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
        return error_covariance
    

    #TODO - H and H_Tranpose is used to match the dimension of error_covariance
    #with the dimension of kalman gain?
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


    def run(self):
        model = Model()
        #kalman gain
        kalman_gain = 0
        measurements = [[4000,280],[4260,282],[4550,285],[4860,286]] 
        position_error = 20
        velocity_error = 5
        position_error_measure = 25
        velocity_error_measure = 6
        acceleration = 2
        R = [[position_error_measure * position_error_measure,0],[0,velocity_error_measure * velocity_error_measure]]
        delta_t = 1
        #A, I, H and H_Tranpose are different if more variables are involved
        A = [[1, delta_t],[0,1]]
        B = [0.5 * delta_t * delta_t, delta_t]
        A_Tranpose = [[1,0],[1,1]]
        #identity matrix
        I = [[1,0],[0,1]]
        H = [[1,0],[0,1]]
        H_Tranpose = [[1,0],[0,1]]
        
        initial_error_covariance = model.cal_initial_error_covariance(position_error,velocity_error) 
        #iteration1
        state_predicted = model.cal_state_predicted(A, measurements[0], B, acceleration)
        error_covariance_predicted = model.cal_error_covariance_predicted(A, initial_error_covariance, A_Tranpose)
        kalman_gain = model.cal_kg(error_covariance_predicted, H, H_Tranpose, R)
        adjusted_state = model.cal_adjusted_state(state_predicted, kalman_gain, measurements[1])
        adjusted_error_covariance = model.cal_adjusted_error_covariance(error_covariance_predicted, kalman_gain, I, H)
        #iteration2
        state_predicted2 = model.cal_state_predicted(A, adjusted_state, B, acceleration)
        error_covariance_predicted2 = model.cal_error_covariance_predicted(A, adjusted_error_covariance, A_Tranpose)
        kalman_gain2 = model.cal_kg(error_covariance_predicted2, H, H_Tranpose, R)
        adjusted_state2 = model.cal_adjusted_state(state_predicted2, kalman_gain2, measurements[2])
        adjusted_error_covariance2 = model.cal_adjusted_error_covariance(error_covariance_predicted2, kalman_gain2, I, H)
        #iteration3
        state_predicted3 = model.cal_state_predicted(A, adjusted_state2, B, acceleration)
        error_covariance_predicted3 = model.cal_error_covariance_predicted(A, adjusted_error_covariance2, A_Tranpose)
        kalman_gain3 = model.cal_kg(error_covariance_predicted3, H, H_Tranpose, R)
        adjusted_state3 = model.cal_adjusted_state(state_predicted3, kalman_gain3, measurements[3])
        adjusted_error_covariance3 = model.cal_adjusted_error_covariance(error_covariance_predicted3, kalman_gain3, I, H)

        #measurements plot
        plt.plot([0, 1, 2, 3], [measurements[0][0], measurements[1][0], measurements[2][0], measurements[3][0]], color = 'red')
        #prediction plot
        plt.plot([0, 1, 2, 3], [measurements[0][0], state_predicted[0], state_predicted2[0], state_predicted3[0]], color = 'blue')
        #adjusted value plot
        plt.plot([0, 1, 2, 3], [measurements[0][0], adjusted_state[0], adjusted_state2[0], adjusted_state3[0]], color = 'green')
        plt.show()


if __name__ == "__main__":
    m = Model()
    m.run()
