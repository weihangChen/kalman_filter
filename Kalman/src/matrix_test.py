import numpy as np
from kalman_filer_youtube import *
#this file test matrix operations and all methods from kalman_filter_impl.py
def test_matrix_basic():
    #matmul test
    matrix_a = [[0.3, 0],[0, 0.291]]
    matrix_b = [[1,0],[0,1]]
    multiply_result = np.matmul(matrix_a, matrix_b)
    assert np.array_equal(multiply_result,[[0.3, 0],[0, 0.291]]) == True
    #subtraction test
    subtraction_result = np.subtract(matrix_b, matrix_a)
    assert np.array_equal(np.round(subtraction_result,1),[[0.7, 0],[0, 0.7]]) == True
    #matmul test again
    matrix_c = [[267.8, 0],[0, 14.8]]
    multiply_result = np.matmul(subtraction_result, matrix_c)
    assert np.array_equal(np.round(multiply_result,1), [[187.5,0],[0,10.5]]) == True
    #2 by 2 matrix multiply with 2 by 1
    matrix_a = [[0.405,0],[0,0.410]]
    matrix_b = [-21,0]
    result = np.round(np.matmul(matrix_a, matrix_b),1)
    assert np.array_equal(result, [-8.5, 0])

def test_cal_state_predicted():
    model = Model()
    delta_t = 1
    A = [[1, delta_t],[0,1]]
    state_predicted_current = [4000,280]
    B = [0.5 * delta_t * delta_t, delta_t]
    acceleration = 2
    state_predicted_next = model.cal_state_predicted(A, state_predicted_current,B, acceleration)
    assert np.array_equal(state_predicted_next, [4281, 282])

def test_cal_initial_error_covariance():
    model = Model()
    initial_covariance_matrix = model.cal_initial_error_covariance(20,5)
    assert np.array_equal(initial_covariance_matrix, [[400,0],[0,25]])
    
def test_cal_error_covariance_predicted():
    model = Model()
    delta_t = 1
    A = [[1, delta_t],[0,1]]
    A_Tranpose = [[1,0],[1,1]]
    P = [[400,0],[0,25]]
    error_covariance_predicted = model.cal_error_covariance_predicted(A, P, A_Tranpose)
    assert np.array_equal(error_covariance_predicted,[[425,0],[0,25]])

def test_cal_kg():
    model = Model()
    H = [[1,0],[0,1]]
    error_convarince = [[425,0],[0,25]]
    H_Tranpose = [[1,0],[0,1]]
    R = [[625,0],[0,36]]
    kg = model.cal_kg(error_convarince, H, H_Tranpose, R)
    assert np.array_equal(kg,[[0.405,0],[0,0.410]])

def test_cal_adjusted_state():
    model = Model()
    state_estimate = [4281, 282]
    K = [[0.405,0],[0,0.410]]
    state_measured = [4260,282]
    adjusted_state = model.cal_adjusted_state(state_estimate, K, state_measured)
    assert np.array_equal(np.around(adjusted_state,1), [4272.5,282])

def test_cal_adjusted_error_covariance():
    model = Model()
    I = [[1,0],[0,1]]
    K = [[0.405,0],[0,0.41]]
    H = [[1,0],[0,1]]
    error_convariance_predicted = [[425,0],[0,25]] 
    adjusted_error_convariance = model.cal_adjusted_error_covariance(error_convariance_predicted, K, I, H)
    assert np.array_equal(np.round(adjusted_error_convariance,1) , [[252.9,0],[0,14.8]])



if __name__ == "__main__":
    try:
        #matrix operator test
        test_matrix_basic()
        #as shown in the video, there are 8 steps in each iteration
        #each test method below contains the end to end method testing against
        #the logic in one step
        #step1 - get [4280, 280] as predicted state
        test_cal_state_predicted()
        #step2 - do once, get [[400,0],[0,25]]
        test_cal_initial_error_covariance()
        #step3 - calculate error convariance for next step using
        #[[400,0],[0,25]] generated from test_cal_initial_covariance
        #result is [[425,0],[0,25]]
        test_cal_error_covariance_predicted()
        #step4 - calculate kalman gain, it consumes the [[425,0],[0,25]]
        #from test_cal_error_covariance_predicted, get [[0.405,0],[0,0.410]]
        test_cal_kg()
        #step5 - calculate the new observation state
        #which is predefined, we skip it here
        #step6 - calculate adjusted sate based on the updated
        #kalman gain + observed state + predicted state
        test_cal_adjusted_state()
        #step7 - update the error covarince
        test_cal_adjusted_error_covariance()
        print("ALL TEST PASS!")
    except Exception as e:
        print('ERROR!!!!!!!!! check stracktrace')
