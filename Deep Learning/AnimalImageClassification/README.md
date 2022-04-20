Neural networks reflect the behavior of the human brain, allowing computer programs to recognize patterns and solve common problems in 
the fields of AI, machine learning, and deep learning. 
<img width="582" alt="image" src="https://user-images.githubusercontent.com/31846843/164258302-b0b0b800-517c-401c-adc7-10112ea48f56.png">

There are different types of NeuralNetworks each of them meant for a specific purpose, CNNs are primarily used for image classification,
object detection and clustering . They are used for spatial data analysis, computer vision, natural language processing, signal processing, 
and various other purposes  CNN accepts images as input. The pixels in the given input image are represented in the numeric color coding 
format RGB(ranging between 0 to 255) and are given to input layers. At each layer weights and bias are introduced. 
The calculated weighted input at each layer is passed as a parameter to the activation function.

Like any other types of neural networks, CNN too has input, output and hidden layers. 
The hidden layers of a CNN typically consist of convolutional layers, pooling layers, fully connected layers, and normalization layers. 
<img width="1052" alt="image" src="https://user-images.githubusercontent.com/31846843/164259656-69d2b328-0c6d-4502-b98d-c8acb0aedda4.png">

Fig1 : Neural network (https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53 (Links to an external site.))

Convolution layer as represented by its name convolutes(combines or intertwines) the image. The process of convolution reduces the size of the image by bringing all the information in the field together into a single pixel.

From the rgb representation of the original image, a subset of pixels (called kernel/filter) where the actual object is present is taken. 
The kernel is traversed with a certain stride across the image. The dot product of kernel and actual rgb values, a simpler representation called a 
feature map is produced. This process is repeated and the resultant feature maps are stacked and passed on as input to the pooling layer. 
The high level features are extracted in this phase.

<img width="758" alt="image" src="https://user-images.githubusercontent.com/31846843/164259930-0d4d8373-cf22-4656-a8f8-60e1a1e42d62.png">

Pooling layer too reduces spatial size of the Convolved Feature. 
From the convoluted representation, the maximum / minimum /average value from the chosen portion of the image is returned.

<img width="562" alt="image" src="https://user-images.githubusercontent.com/31846843/164260080-922e57a1-d374-43b7-9c70-5f31650c439c.png">

Activation functions are a critical part of the design of a neural network and determine how well the model learns on training data. 
The choice of activation function in the output layer will define the type of predictions the model can make. 
Sigmoid, Relu and TanH are the most used ones. The activation functions introduced the non-linearity to the model by compressing 
the input and returning the output in the range bound [0-1] in case of sigmoid function, [0-infinity] in case of relu and [-1,1] in 
case of sigmoid function.

Upon passing through the convolution and the pooling layers, the learning process of the model completes. The image shrinks in size after
passing through a series of convolution and pooling layers and finally reaches the fully connected layer.

<img width="814" alt="image" src="https://user-images.githubusercontent.com/31846843/164260386-50db78ac-2946-4e89-a742-6d6d823ee86d.png">

Fully Connected Layer is simply a feed-forward network. The image is flattened and for each of the layer, the following calculation is performed.

g(Wx+b)

W = weight at each layer, x = input vector, b= bias, g= activation function.

Finally after reaching the output layer, the softmax activation function is applied to predict the probability of the classifying the image.

https://medium.datadriveninvestor.com/introduction-to-how-cnns-work-77e0e4cde99b (Links to an external site.)

https://medium.com/analytics-vidhya/understanding-neural-networks-from-neuron-to-rnn-cnn-and-deep-learning-cd88e90e0a90 (Links to an external site.)

https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53 (Links to an external site.)

https://towardsdatascience.com/convolutional-neural-network-17fb77e76c05 (Links to an external site.)


