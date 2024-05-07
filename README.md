# Particle Swarm-Optimized Artificial Neural Network for Non-Invasive Glucose Measurement and HbA1c Computation
In this work, a particle-swarm optimization-based artificial neural network for non-invasive continuous glucose monitoring using the principles of near-infrared spectroscopy (NIRS) is proposed. It is shown that the PSO-ANN approach outperforms the traditional backpropagation algorithm used in ANN training and several other regression algorithms with the lowest error metrics: ```MAE- 1.01```, ```MSE-2.16```, ```RMSE-0.97```, ```𝑅-squared-0.976``` and ```modified 𝑅-squared-0.973```. The 3-stage methodology adopted in this work is shown below.

![alt text](https://github.com/rdharini2001/Non-Invasive-Glucometer/blob/main/graphical_gluco.JPG)

The accuracy and reliability of the proposed system are analysed using the Clarke Error Grid (CEG) with 93.9% of the obtained readings falling within zone A and 100% of the readings falling in the clinically accepted range (zones A and B). Refer to the [preprint](https://www.techrxiv.org/doi/full/10.36227/techrxiv.24465955.v1) for more details. 

# Data Format
Inputs:
```BMI - np.array(b1, b2, b3,.....bn)```
```Voltage (in mV) - np.array(v1, v2, v3,......vn)```
```Age - np.array(a1, a2, a3,......an)```
Output:
```glucose in mg/dl```

Run ```pso_ann.py``` after mosdifying ```Xtrain``` and PSO parameters.

Please consider citing the work if you find it useful. 
```
@article{Particle Swarm-Optimized Artificial Neural Network for Non-Invasive Glucose Measurement and HbA1c Computation,
  author = {Suma KV, Dharini Raghavan, Maya V Karki, Narayana Sharma and Gundu Rao},
  doi = {10.36227/techrxiv.24465955.v1},
  journal = {Techrxiv},
  pages = {1--4},
  title = {{Particle Swarm-Optimized Artificial Neural Network for Non-Invasive Glucose Measurement and HbA1c Computation}},
  year = {2023}
}
```
