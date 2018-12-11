class Model:
    
    def cal_kg(self, error_est, error_mea):
        kg = error_est / (error_est + error_mea)
        return round(kg,2)

    def cal_estimate(self, est_before, kg, mea):
        est_now = est_before + kg*(mea - est_before)
        return round(est_now,2 )

    def cal_error(self, kg, error_est_before):
        error_est_now = (1-kg)*error_est_before
        return round(error_est_now,2)


   
    #https://bit.ly/2C06f0Q single step
    #https://bit.ly/2QK69Tw multi steps
    #this method will verify against conent for 2 iterations from the second video
    #the updated values are "est", "error_est" and "kg"
    def run(self):
        #const and init values
        error_mea_const = 4
        mea1 = 75
        mea2 = 71
        #values that are updated continously
        error_est = 2 
        kg = 0
        est = 68
        

        #first iteration
        kg = self.cal_kg(error_est, error_mea_const)
        assert kg == 0.33
        #estimate current step
        est = self.cal_estimate(est, kg, mea1)
        assert est == 70.31
        #error current step
        error_est = self.cal_error(kg, error_est)
        assert error_est == 1.34

        #second iteration
        kg = self.cal_kg(error_est, error_mea_const)
        assert kg == 0.25
        est = self.cal_estimate(est, kg, mea2)
        assert est == 70.48
        error_est = self.cal_error(kg, error_est)
        assert error_est == 1.01
        

#demo1.py contains the most simplified version of implementation, no matrix
if __name__ == "__main__":
    m = Model()
    m.run()
    print("PASS!")