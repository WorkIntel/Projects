"""
% Find sub-sample location of a global peak within 2D-matrix by applying 
% two dimensional polynomial fit & extremum detection. 
%
% Sample usage: 
% >> M = exp(-((1:30) - 19.5).^2/(2*5^2)); % gauss: center=19.5; sigma=5
% >> P = peakfit2d(M'*M);                  % find peak in 2D-gauss
% >> disp(P);
%   19.5050   19.5050
%
% Algebraic solution derived with the following steps:
%
% 0.) Define Approximation-Function: 
%
%     F(x,y) => z = a*x^2+b*x*y+c*x+d+e*y^2+f*y
%
% 1.) Formulate equation for sum of squared differences with
%
%     x=-1:1,y=-1:1,z=Z(x,y)
%
%     SSD = [ a*(-1)^2+b*(-1)*(-1)+c*(-1)+d+e*(-1)^2+f*(-1) - Z(-1,-1) ]^2 + ...
%              ...
%             a*(+1)^2+b*(+1)*(+1)+c*(+1)+d+e*(+1)^2+f*(+1) - Z(-1,-1) ]^2
%        
% 2.) Differentiate SSD towards each parameter
%
%     dSSD / da = ...
%              ...
%     dSSD / df = ...
%
% 3.) Solve linear system to get [a..f]
%
% 4.) Differentiate F towards x and y and solve linear system for x & y
%
%     dF(x,y) / dx = a*... = 0 !
%     dF(x,y) / dy = b*... = 0 !
"""

import numpy as np
import matplotlib.pyplot as plt

def peak_fit_2d(Z):
    """
    Finds the sub-pixel location of a global peak within a 2D matrix using a polynomial fit.

    Args:
        Z: The input 2D matrix.

    Returns:
        A tuple (yp, xp) representing the sub-pixel peak location.
    """

    sZ = Z.shape
    if np.min(sZ) < 2:
        print('Wrong matrix size. Input matrix should be numerical MxN type.')
        return (0, 0)

    # Find global maximum and extract 9-point neighborhood
    v, p = np.max(Z), np.argmax(Z)
    yp, xp = np.unravel_index(p, sZ)

    if yp == 0 or yp == sZ[0] - 1 or xp == 0 or xp == sZ[1] - 1:
        print('Maximum position at matrix border. No subsample approximation possible.')
        return (yp, xp)

    K = Z[yp - 1:yp + 2, xp - 1:xp + 2]

    # Approximate polynomial parameters
    a = (K[1, 0] + K[0, 0] - 2 * K[0, 1] + K[0, 2] - 2 * K[2, 1] - 2 * K[1, 1] + K[1, 2] + K[2, 0] + K[2, 2])
    b = (K[2, 2] + K[0, 0] - K[0, 2] - K[2, 0])
    c = (-K[0, 0] + K[0, 2] - K[1, 0] + K[1, 2] - K[2, 0] + K[2, 2])
    e = (-2 * K[1, 0] + K[0, 0] + K[0, 1] + K[0, 2] + K[2, 1] - 2 * K[1, 1] - 2 * K[1, 2] + K[2, 0] + K[2, 2])
    f = (-K[0, 0] - K[0, 1] - K[0, 2] + K[2, 0] + K[2, 1] + K[2, 2])

    # Calculate sub-pixel shift
    dn =(16 * e * a - 9 * b**2)
    ys = (6 * b * c - 8 * a * f) / dn
    xs = (6 * b * f - 8 * e * c) / dn

    return xs + xp, ys + yp 

def create_zdata(dx = 0.125, dy = 0.125):
    "create 2D data"
    w, h     = 10, 10
    x,y      = np.meshgrid(np.arange(w),np.arange(h))   
    x,y      = x + dx, y + dy
    #z        = w - (np.abs(x - w/2) + np.abs(y - h/2))  # less slope
    z        = w - (np.abs(x - w/2)**2 + np.abs(y - h/2)**2)  # less slope
    #z        = w - (np.abs(x - w/2)**4 + np.abs(y - h/2)**4) 
    return z

def show_3d(z):
    "show 3d data"
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    w, h     = z.shape
    x,y      = np.meshgrid(np.arange(w),np.arange(h))      
    ax.scatter(x, y, z, marker='.')

    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    #ax.set_aspect('equal', 'box')
    plt.show() #block=False)        


if __name__ == '__main__':
    
    shifts   = [(0,0),(-0.5, 0.5), (0.5, 0.5), (-0.3, -0.3), (0.333, -0.333), (0.145, -0.0)]
    for shift in shifts:
        xs, ys   = shift
        z        = create_zdata(xs, ys)
        xp, yp   = peak_fit_2d(z)
        print(xs, ys, xp, yp)

        show_3d(z)
   