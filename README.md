# NumpyNeuralNet

As [the man himself](https://en.wikipedia.org/wiki/Richard_Feynman) said "What I cannot create, I do not understand.".  
So I try to built Neural Networks and training algorithms using only Numpy.

## Densely connected NN
    
Train a MNIST model with [output_activation]_[Loss].

-   Sigmoid - MSE
    ```shell
    $ python test/dense.py sigmoid_mse train
    ```

-   Sigmoid - Cross-entropy
    ```shell
    $ python test/dense.py sigmoid_crossentropy train
    ```

-   Softmax - Log-likelihood
    ```shell
    $ python test/dense.py softmax_loglikelihood train
    ```
