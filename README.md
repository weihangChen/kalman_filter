# kalman_filter

### Goal
this is the implementation of Kalman filter lectures on 
https://www.youtube.com/watch?v=CaCcOwJPytQ&index=1&list=PLX2gX-ftPVXU3oUFNATxGXY90AULiqnWT

### How to run
there are three python script files, each file can be run seperatly, the file with main logic calculation is in kalman_filter_impl.py
1. temperature.py corresponds to the first few lectures, that walks through the single dimension senario
2. kalman_filter_impl.py contains all the main logics (8 steps process) from the lectures. when script is run, you are expected to see a drawing with three plots.
one for predicted states, one for measurements, one for adjusted states. As shown in the lectures, three interations are done in this implementation as well, and the result are verified. (the calculation for the last itration from the video has some errors, and you should be able to find out)

Inline-style: 
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")

3. matrix_test.py contains some matrix operator tests, plus assertion of the what happens in each step in the complete pipeline.


### differece between this and the requirement from homework
1. the prediction equation is different, because it takes acceleration into account.
2. no noice is taken into account. 
3. the state matrix only contains 2 variables, while requirement has 6. The the only difference should be so 
that the A_Matrix, H and H_Tranpose matrix are different.

