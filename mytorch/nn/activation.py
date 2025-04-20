import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):

     
        """
        :param Z: Data Z s(*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        # Compute the softmax in a numerically stable way
        # Apply it to the dimension specified by the `dim` parameter
        exponential = np.exp(Z)
        sum_by_row = np.sum(exponential, self.dim)
    
        sum_by_row = np.expand_dims(sum_by_row,self.dim)
        tiling_shape = list(Z.shape)
        for i in range(0,len(tiling_shape)):
            
            if i != self.dim:
                tiling_shape[i] = 1
        
        self.A = np.tile(np.power(sum_by_row,-1),tiling_shape)*exponential
        return self.A



    def backward(self, dLdA):

        def old_back(A, dLdA):
            N = A.shape[0]
            C = A.shape[1]

            # Initialize the final output dLdZ with all zeros. Refer to the writeup and think about the shape.
            dLdZ = np.zeros(dLdA.shape)

            # Fill dLdZ one data point (row) at a time.
            for i in range(N):
                # Initialize the Jacobian with all zeros.
                # Hint: Jacobian matrix for softmax is a _×_ matrix, but what is _ here?
                J = np.zeros([C,C])
                row = A[i]
                # Fill the Jacobian matrix, please read the writeup for the conditions.
                for m in range(C):
                    for n in range(C):
                        if m==n:
                            J[m,n] = row[m]*(1-row[m])
                        else:
                            J[m,n] = -row[m]*row[n]

                # Calculate the derivative of the loss with respect to the i-th input, please read the writeup for it.
                # Hint: How can we use (1×C) and (C×C) to get (1×C) and stack up vertically to give (N×C) derivative matrix?
                dLdZ[i] = dLdA[i]@J
            self.dLdZ = dLdZ
            return dLdZ
        

        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        
        # Get the shape of the input
        shape = self.A.shape
        # Find the dimension along which softmax was applied
        C = shape[self.dim]
        A = self.A
        # Reshape input to 2D
        if len(shape) > 2:

            A = np.moveaxis(A,self.dim,-1)
            A = A.reshape(-1,dLdA.shape[-1])

            dLdA = np.moveaxis(dLdA,self.dim,-1)
            dLdA = dLdA.reshape(-1,dLdA.shape[-1])
            
            dLdZ = old_back(A,dLdA)
        
        return dLdZ.reshape(self.A.shape)

        
 

    