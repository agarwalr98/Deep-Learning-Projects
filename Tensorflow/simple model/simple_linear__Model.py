#import 
# %matplotlib inline
import matplotlib.pyplot as plt 
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

#Load Data
from mnist import MNIST
data = MNIST(data_dir="data/MNIST/")

# print the size of dataset
print("Size of :")
print("- Training-set:\t\t{}".format(data.num_train))
print("- Validation-set:\t{}".format(data.num_val))
print("- Test-set:\t\t{}".format(data.num_test))


#The images are stored in one-dimensional arrays of this length.
img_size_flat = data.img_size_flat

#Tuple with height and width of images used to reshape arrays.
img_shape = data.img_shape

#Number of classes , one class for each of 10 digits.
num_classes = data.num_classes

# one hot encoding representation
data.y_test[0:5,:]

# integer class numbers 
data.y_test_cls[0:5]


# function to plot the images
# Arguments passed are images with true class label
def plot_images(images,cls_true,cls_pred=None):
  assert len(images) == len(cls_true)==9
  
  #create figure with 3*3 sub -plots
  fig,axes = plt.subplots(3,3)
  fig.subplots_adjust(hspace = 0.3 , wspace = 0.3)
  
  for  i , ax in enumerate(axes.flat):
      #plot image
      ax.imshow(images[i].reshape(img_shape),cmap="binary")
      
      #show true and predicted classes.
      if cls_pred is None:
        xlabel = "True : {0}".format(cls_true[i])
      else:
        xlabel = "True : {0},pred : {1}".format(cls_true[i],cls_pred[i])
       
      ax.set_xlabel(xlabel)
      
      #Remove ticks from the plot.
      ax.set_xticks([])
      ax.set_yticks([])
    
   #Ensure that plot is shown correctly
  plt.show()




# cross check images by plotting it
# Get the first images from the test-set.
images = data.x_test[0:9]

# Get the true classes for those images.
cls_true = data.y_test_cls[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true)



#Placeholder variable
x= tf.placeholder(tf.float32,[None,img_size_flat])

y_true = tf.placeholder(tf.float32,[None , num_classes])

y_true_cls = tf.placeholder(tf.int64,[None])

# variables which needs to be optimized during training
weights = tf.Variable(tf.zeros([img_size_flat,num_classes]))

# one dimensional tensor called as biases of length num_classes
biases = tf.Variable(tf.zeros([num_classes]))


logits = tf.matmul(x,weights) + biases

y_pred  = tf.nn.softmax(logits)

y_pred_cls = tf.argmax(y_pred,axis = 1)

# performance measure used in classification
# To see how well our parameteres are optimized
# it is defined for single image
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels = y_true)

# average of cross-entropy  for all image
cost = tf.reduce_mean(cross_entropy)

#optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.5).minimize(cost)

#Performance measure
correct_prediction = tf.equal(y_pred_cls,y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#Tensorflow run
#to execute the graph.
session = tf.Session()

session.run(tf.global_variables_initializer())

# number of images taken at a time
batch_size = 100

def optimize(num_iterations):
  for i in range(num_iterations):
      #Get a batch of training examples.
      #x_batch now holds a batch of images and 
      #y_true_batch are the true labels of those batch images.
      x_batch ,y_true_batch ,_ = data.random_batch(batch_size= batch_size)
      
      #put the batch into a dict with the proper names
      #for placeholder variables in the Tensorflow graph.
      #Note that the plceholder for y_true_cls is not set #
      #because it is not used during training.
      feed_dict_train = {x:x_batch,
                        y_true : y_true_batch}
      
      #Run the optimizer using this batch of training data .
      #TensorFlow assigns the variables in feed_dict_train
      #to the placeholder variables and then runs the optimizer.
      session.run(optimizer,feed_dict_train)

feed_dict_test = {
      
            x:data.x_test,
    y_true:data.y_test,
    y_true_cls: data.y_test_cls
}

def print_accuracy():
    # Use TensorFlow to compute the accuracy.
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    
    # Print the accuracy.
    print("Accuracy on test-set: {0:.1%}".format(acc))

def print_confusion_matrix():
    #Get the true classification for the test-set.
    cls_true = data.y_test_cls
    
    #Get the predicted classification for the test-set.
    cls_pred = session.run(y_pred_cls,feed_dict = feed_dict_test)
    
    #Get the confusion matrix using sci-kitlearn
    cm = confusion_matrix(y_true=cls_true,
                          y_pred = cls_pred
                         )
    
    #print the confusion matrix as text.
    print(cm)
    
    #plot the confusion matrix as an image.
    plt.imshow(cm,interpolation ='nearest',cmap=plt.cm.Blues)
    
    #Make various adjustments to the plot.
    plt.tigth_layout()
    plt.colorbar()
    tick_marks = np.arrange(num_classes)
    plt.xticks(tick_marks,range(num_classes))
    plt.yticks(tick_marks,range(num_classes))
    plt.xlabel('predicted')
    plt.ylabel('True')
    
    
    plt.show()

def plot_example_errors():
    #Use TensorFlow to get a list of boolean values
    #Whether each test image has been correctly classified,
    #and a list for the predicted class of each image.
    correct , cls_pred = session.run([correct_prediction,y_pred_cls],
                                    feed_dict = feed_dict_test)
    #Negate the boolean array.
    incorrect = ( correct == False)
    
    #Get the images from the test-set that have been
    #incorrectly classified.
    images = data.x_test[incorrect]
    
    #Get the predicted classes for those images .
    cls_pred = cls_pred[incorrect]
    
    #Get the true classes for those images 
    cls_true = data.y_test_cls[incorrect]
    
    #plot the first 9 images 
    plot_images(images=images[0:9],
                cls_true= cls_true[0:9],
                cls_pred=cls_pred[0:9])

def plot_weights():
    #Get the values for the weights from the TensorFlow variable.
    w = session.run(weights)
    
    #Get the lowest and highest values for the weights.
    # This is used to correct the color intensity across the 
    #images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)
    
    #create figure with 384 subplots
    #where the last 2 subplots are unused.
    fig,axes = plt.subplots(3,4)
    fig.subplots_adjust(hspace=0.3,wspace=0.3)
    
    for i ,ax in enumerate(axes.flat):
        #Only use the weights for the first 10 sub-plots.
        if i<10 :
            image = w[:,i].reshape(img_shape)
            
            #Set the label for the sub-plot.
            ax.set_xlabel("weights: {0}".format(i))
            
            #plot the image 
            ax.imshow(image , vmin=W_min,vmax=w_max,cmap='seismic')
         # Remove ticks from each sub plot
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

for i in range(1,28):
  
  optimize(num_iterations =i)
  print_accuracy()



plot_example_errors()