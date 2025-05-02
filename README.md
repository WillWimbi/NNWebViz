This is a neural network trainer and visualizer for the MNIST dataset.

The site can be found at [wimbiscus.com](https://wimbiscus.com)

The user can train a neural network on the MNIST dataset and visualize the training process. They can observe activations, gradients, weights, loss and validation loss. 
They can also adjust the hyper/params of the network: the learning rate, choosing the optimizer, and notably adjusting the convolution filter sizes which are currently guaranteed to work.
The user can observe the actual image classifications performed by the network at each stage (ALL on the validation set, no performance fudging) and can observe the top three predictions for a digit.
This allows one to observe what kinds of inputs the network handles well and struggles with, based on its current architecture and training data. 
Additionally, the user can navigate to a second page to observe the comparisons across 50 networks with identical architectures trained on the same data, 
and observe the images across this sampleset that were most misclassified.

Bonuses: - For the first five days I attempted to build a neural network from scratch in javascript (neuralNetMnist.js), doing the low level matmul operations myself in arrays. 
I eventually learned it would've been far better to start with flat arrays and build up from there, but upon observing the complexity and the need to provide a working implementation I decided to use Tensorflow.js, although that certainly didn't make the project trivial. 
- runStaticTrainingTests.js was used to create the 50 identical networks for the second viewing page.

Annoying issue: Due to the excessive use of await and async functions when rendering the images, there is currently a noticeable waterfall effect when the training first starts for the activations/gradients/weights visualizations.
This would be a nice fix however the project has gotten incredibly long so I figured I'd leave it as is. The project still achieves its goals of visualization and training experience. 

For who is reading this this:
 - **userTrainingExperience.js** is the *overwhelming* bulk of the work, what I mainly focused on (after I stopped working on the vanilla JS net from scratch attempt).
 - index.html is main page.
 - pretrainedNets.html is the pretrained net visualization page.
 - userPretrainedNetsExperience.js is the logic for loading and visualizating the pretrained nets.
 - run.js in /API is used for my **VERCEL DEPLOYMENT**. I used **MONGO DB** for the pretrained network info.
My files are indeed somewhat messy. userTrainingExperience.js which took the longest to complete (how to render successfully log full activations/gradients/weights and TF undefined/datasync hell AHHH before working it all out)
