This explains the development of an MPC algorithm for outer-loop control of a quadrotor.
It is based on CasADi optimization framework coded in MATLAB. 
An extended Kalman Filter (EKF) is utilized for state estimation.

The quadrotor is initialized and commanded to go to [5,5,5]m in the inertial frame. The evolution of the states is given in the Figure below.

![MPC_EKF_states](https://github.com/user-attachments/assets/a2739d74-ba28-4683-9b18-0edc4dec4406)

The evolution of the control inputs is given below.

![MPC_EKF_inputs](https://github.com/user-attachments/assets/cbe50bf5-5592-45e5-9241-a272b48751b5)

