#NNet is a neural network library written in C++. A list of methods available to the NNet object and thier brief descriptions is given below:


##Standard Methods:

`NNet.init(string sconfig, int iclassreg, int inumcores, int igradd, int icostfunc, int epoch = -1)`

This method is used to initilize the neural network. The architecture is specified by the string `sconfig` with `-` separated numbers. NNet automatically detects the number of input and output layers depending on the input file, thus only the architecture of the hidden layers needs to be specified for sconfig. The user is required to specify what would be the purpose of the neural network, regression or classification. The variable `iclassreg` should be set to 1 for regression and 0 for classification. The user can specify the number of cores they have at thier disposal, if the number of cores is greater than 1 NNet would use that to its advantage to perform parallel calculations. The user can also decide what gradient descent algorithim they prefer to use, set `igradd` to 0 for batch gradient descent and 1 for stochastic gradient descent. For classification the user can specify NNet to use the cross entropy cost function which can be done by specifying `icostfunc` to be 1 (not available yet). If the user has selected stochastic gradient descent they can specify the number of epoch to perform using the `epoch` variable


`NNet.func_arch(string flayer)`

The user can specify a very specific combinations of activation functions for the network by specifying thier choice as a string. E.g
`NNet.func_arch("030")` would use the sigmoid function for the first hidden layer, a tanh + linear function for the second hidden layer and sigmoid again for the output layer. The list of available activation functions and thier corresponding numeric values are:
* `0`: Sigmoid function
* `1`: Tangent Hyperbolic
* `2`: Rectified linear
* `3`: Tangent Hyperbolic + Linear


`NNet.load(string filename, int imode = 0, string sep1 = ",", string sep2 = " ")`

This method is used to load the file into the network. The filename is specified by `filename`. If `imode` is 0 then the data is split into training, test and validation data, if `imode` is 1 then the entire file is used for training. This method assumes that the file contains both input and output data where each component of the input and output vector is seperated by the string specified in `sep1` and the input and output vectors are seperated by the string specified in sep2. By default sep1 is a comma (",") and sep2 is a space (" ").


`NNet.train_net(double lrate, int mode = 0)`

This methods trains the neural network using standard backpropogation, the learning rate is specified by the variable `lrate`. If `mode` is set to 1 it gives the current accuracy and/or RMSE of the neural net on the given data set.


`NNet.train_rprop(int mode = 0, double tmax = 15.0)`

This method trains the neural network using resilient backpropogation. If `mode` is set to 1 it prints the RMSE and/or accuracy as the net is trained. The variable `tmax` sets an upper bound on the amount by which a particular weight can change


`NNet.test_file(string filename, int ffmode = -1, string sep1 = ",", string sep2 = " ")`

This method allows the user to upload a file to test the neural network against. `ffmode` is best left to its value of -1, else non-optimal weights and biases would be used to test the file.


`NNet.test_net(int mode = 0)`

This method allows the user to test thier neural network on the file uploaded in load. If `mode` is set to one then the neural network is tested on the validation set.


`NNet.savenet(string netname)`

Save the given network with the a name specified in `netname`.


`NNet.loadnet(string netname)`

Load the neural network saved with the name specified in `netname`.


`NNet.snets(void)`

View all saved neural networks in the given directory.

