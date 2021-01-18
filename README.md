# Neural Style Transfer
---

This repository transfers a content image into an artwork style by applying the idiosyncracies of a given style image into the content image.

For example:- 

### Content Image

![content_image](images/content/my_pic_compressed.jpg "content-image" )

### Style Image

![style_image](images/style/sn.jpg "style-image")

### Final Output

![Final output](nst_output/my_pic_starry-night-van-gogh.jpg "final-output")

---

# Steps for Interacting with the Repo

1. In your git bash, run the following command :-

		git clone https://github.com/kunalyadav0954/Neural_style_transfer.git
		cd Neural_style_transfer
		
2. Save your content image in the folder `images/content/` and the style image in the folder `images/style/` 

3. Download the pretrained vgg-19 model from the following link : [imagenet-vgg-19-verydeep.mat](https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat "Pretrained-model") and save it in a folder named `vgg-19_pretrained_model` (Create this folder in the Neural_ style_ transfer directory)

4. In an IDE of your choice, simply import the module nst.py as follows:

		import importlib  # for importing nst again without restarting IDE
		import nst
		 
5. The final out put will be stored in the folder `nst_output`You can tweak the parameters like, content, style images, style layers, no. of iterations, content layer etc in the CONFIGURE class defined at the start of nst.py :

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