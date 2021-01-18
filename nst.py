# Performs neural style transfer from the specified style image to the specified content image

print('Importing required libraries.......')
from nst_utils import *
import tensorflow as tf
import matplotlib.pyplot as plt
print('Libraries imported successfully')


class CONFIGURE:
  content_path ='images/content/my_pic.jpg'
  style_path ='images/style/rembrandt.jpg'
  STYLE_LAYERS = [
    ('conv1_2', 0.7),
    ('conv2_2', 0.2),
    ('conv3_4', 0.05),
    ('conv4_4', 0.025),
    ('conv5_4', 0.025)]
  # use last layers for style
  content_cost_layer = 'conv4_4'   # conv4_4
  num_iterations = 1001
  learning_rate = 1.0
  output_folder = 'nst_output/'


def content_cost(a_C, a_G):
    """
    Computes content cost for the generated image wrt to the content image
    :param a_C: hidden layer activation for the content image (tensor, shape: (1,n_H, n_W, n_C)
    :param a_G: hidden layer activation for the generated image (tensor, shape: (1,n_H, n_W, n_C)
    :return:
    content_cost : ( 1/(4*n_H*n_W*n_C) ) * summation (a_c - a_G)**2  (tensor type)
    """
    # retrieve layer dimensions :
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # unrolling vectors a_C and a_G :
    a_C = tf.transpose(tf.reshape(a_C, shape=[m, n_H * n_W, n_C]), perm=[0, 2, 1])
    # ^ will be converted to shape [m, n_C, n_H*n_W]

    a_G = tf.transpose(tf.reshape(a_G, shape=[m, n_H * n_W, n_C]), perm=[0, 2, 1])

    J_content = tf.multiply(tf.reduce_sum(tf.square(tf.subtract(a_C, a_G))), 1 / (4 * n_H * n_W * n_C))

    return J_content


def gram_matrix(A):
    """
    Calculates the style/gram matrix for a hidden layer's activations A
    :param A: tensor of shape (n_C, n_H*n_W)
    :return: style matrix G = A.AT (tensor, shape:(n_C,n_C) )
    """
    return tf.matmul(A, tf.transpose(A))


def style_cost(a_S, a_G):
    """
    Computes style cost for generated image w.r.t style image
    :param a_S: hidden layer activations for style image (tensor, shape : (1,n_H, n_W, n_C) )
    :param a_G: hidden layer activations for generated image (tensor, shape : (1,n_H, n_W, n_C) )
    :return: style cost =  [1/ (4 * (n_c**2) * ( (n_H*n_W)**2 ) ) ] * summation( (GG-GS)**2 )
    """

    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # unrolling vectors into shape (n_c, n_H*n_W)
    a_S = tf.reshape(tf.transpose(tf.reshape(a_S, shape=[m, n_H * n_W, n_C]), perm=[0, 2, 1]), shape=[n_C, n_H * n_W])

    a_G = tf.reshape(tf.transpose(tf.reshape(a_G, shape=[m, n_H * n_W, n_C]), perm=[0, 2, 1]), shape=[n_C, n_H * n_W])

    # computing gram/style matrices
    G_G = gram_matrix(a_G)
    G_S = gram_matrix(a_S)

    #computing style cost
    J_style = tf.multiply(tf.reduce_sum(tf.square( tf.subtract(G_S, G_G) ) ), 1/( 4 * (n_C**2) * ((n_H*n_W)**2) )  )

    return J_style

# We'll be using multiple hidden layers for calculating style cost for better results
# differnet weights are assigned to different layers

# og weights : 0.2 each layer
STYLE_LAYERS = CONFIGURE.STYLE_LAYERS


# compute style cost over all the layers specified by STYLE_LAYERS

def total_style_cost(model, STYLE_LAYERS):
    """
    Computes overall style_cost
    :param model: dictionary containing our tensorflow model layers
    :param STYLE_LAYERS: list of tuples of (layer_name, coeff) as defined above this function
    :return: overall style_cost J_style
    """

    J_style = 0  # initializing style_cost

    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out
        J = style_cost(a_S, a_G)
        J_style += coeff * J

    return J_style

def total_cost(J_content, J_style, alpha=10, beta=40):
    return (alpha * J_content) + ( beta * J_style )

# Reset the graph
tf.reset_default_graph()

# Start interactive session (so we can use eval)
sess=tf.InteractiveSession()

# Load our style and content images (also reshaping and normalizing them for our vgg model
content_path = CONFIGURE.content_path
style_path = CONFIGURE.style_path
out_file_name=content_path.split('/')[-1].split('.')[0]+'_'+style_path.split('/')[-1].split('.')[0]
content_image, content_input_to_vgg = load_image(content_path)
style_image, style_input_to_vgg = load_image(style_path)

# Create generated image
generated_image = generate_noise_image(content_input_to_vgg, noise_ratio=0.6)

# Loading pretrained model
print('\n\n------Loading pretrained vgg-19 model--------')
model=load_vgg_model('vgg-19_pretrained_model/imagenet-vgg-verydeep-19.mat')
print('Model loaded successfully')

print('\n\nCreating content_cost graph')
# setting content image as input to the pretrained model
sess.run(model['input'].assign(content_input_to_vgg))
# Using conv4_2 layer for calculating content cost
out = model[CONFIGURE.content_cost_layer]
a_C = sess.run(out) # evaluated for content image
a_G = out  # not yet evaluated for generated image
J_content = content_cost(a_C, a_G)
print('Done')

print('Creating Style Cost graph')
# setting input of the model to our style image
sess.run(model['input'].assign(style_input_to_vgg))
J_style = total_style_cost(model,STYLE_LAYERS)
print('Done')

J_total = total_cost(J_content, J_style)
print('Total cost graph also created')

# Choose an optimizer with a learning rate
optimizer = tf.train.AdamOptimizer(CONFIGURE.learning_rate)  # og rate : 2.0
train_step = optimizer.minimize(J_total)

# Initialize variables
sess.run(tf.global_variables_initializer())
# set generated_noise image as input to our model
sess.run(model['input'].assign(generated_image))
final_iteration_value = CONFIGURE.num_iterations - 1

print('\n\nTraining the model now.....')
for i in range(CONFIGURE.num_iterations):
    sess.run(train_step)
    # print costs for every 20 iterations
    if i%20==0:
        Jt, Jc, Js = sess.run([J_total, J_content, J_style])
        print('-----------Iteration : {}-----------'.format(i))
        print('Content Cost : {}'.format(Jc))
        print('Style Cost : {}'.format(Js))
        print('Total Cost : {}'.format(Jt))
    if i==final_iteration_value:
        painted_image = sess.run(model['input'])  # numpy form (1,300,400,3)
        painted_image = painted_image + CONFIG.means  # un-normalize
        clipped_img = np.clip(painted_image[0],0,255).astype('uint8') # for imshow
        image = Image.fromarray(clipped_img)
        image.save(CONFIGURE.output_folder+'{}.jpg'.format(out_file_name))

# Close the interactive session
sess.close()




#fig, axes= plt.subplots(1,2)
#axes[0].set_title('Content Image')
#axes[0].imshow(content_image)
#axes[1].set_title('Style Image')
#axes[1].imshow(style_image)
