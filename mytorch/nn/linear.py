import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        self.A = A
        og_size = A.shape
        self.batch_size = A.size//A.shape[-1]
        A = A.reshape((self.batch_size,A.shape[-1]))
  
        out_shape = self.W.shape[0]
        new_shape = list(self.A.shape)
        new_shape[-1] = out_shape
        print(new_shape)
        return (A@(self.W.T)+self.b).reshape(new_shape)

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # TODO: Implement backward pass
        dLdZ = dLdZ.reshape((self.batch_size,dLdZ.shape[-1]))
        new_A = self.A.reshape((self.batch_size,self.A.shape[-1]))

        # Compute gradients (refer to the equations in the writeup)
        self.dLdA = dLdZ@self.W
        self.dLdW = dLdZ.T@new_A
        self.dLdb = np.sum(dLdZ,0)
        
        
        # Return gradient of loss wrt input
        return self.dLdA.reshape(self.A.shape)
