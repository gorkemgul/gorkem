# While creating a project creating functions is really invaluable, but since we build many projects day by day,
# I personally think that it's a good idea to create a script to call our functions instead of rewriting them all the time.

import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

# Create a function to show an image
def imshow(image, title='', size=8):
    """
    Takes an image, turns it into RGB and shows it.

    Args:
      image: an image
      title: name of the image
      size = a number to calculate the figsize

    Returns:
      an image in RGB format
    """
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w / h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title);


# Define a translation function
def image_translation(image_path=''):
    """
    Takes an image path loads the image and translates it.

    Args:
      image: the image path to upload and translate

    Returns:
      A Translated image
    """
    # Load the image and store its height and width
    untranslated_image = cv2.imread(image_path)
    height, width = untranslated_image.shape[:2]

    # Shift it by quarter of the height and width
    quarter_height = height / 4
    quarter_width = width / 4

    # Define translation matrix as T
    T = np.float32([[1, 0, quarter_width], [0, 1, quarter_height]])

    # Use warpAffine function to transform the image using our defined translation matrix (T)
    translated_image = cv2.warpAffine(untranslated_image, T, (width, height))

    # Plot our untranslated image
    plt.figure(figsize=(16, 9))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(untranslated_image, cv2.COLOR_BGR2RGB))
    plt.title('Non-Translated Image')
    plt.axis(False)

    # Plot our translated image
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(translated_image, cv2.COLOR_BGR2RGB))
    plt.title('Translated Image')
    plt.axis(False)
    plt.show()

# Define a rotation function
def image_rotation(image_path, scale):
    """
    """
    # Read the image and take its height and width
    unrotated_image = cv2.imread(image_path)
    height, width = unrotated_image.shape[0], unrotated_image.shape[1]

    # Define the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 90, scale=scale)

    # Use warpAffine function to rotate the image using our defined rotation matrix
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    # Plot our untranslated image
    plt.figure(figsize=(16, 9))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(unrotated_image, cv2.COLOR_BGR2RGB))
    plt.title('Non-Rotated Image')
    plt.axis(False)

    # Plot our translated image
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
    plt.title('Rotated Image')
    plt.axis(False)
    plt.show()

# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_image(filename, img_shape=224, scale=True):
  """
  Reads in an image from filename, turns it into a tensor and reshapes into
  (224, 224, 3).
  Parameters
  ----------
  filename (str): string filename of target image
  img_shape (int): size to resize target image to, default 224
  scale (bool): whether to scale pixel values to range(0, 1), default True
  """
  # Read in the image
  img = tf.io.read_file(filename)
  # Decode it into a tensor
  img = tf.image.decode_jpeg(img)
  # Resize the image
  img = tf.image.resize(img, [img_shape, img_shape])
  if scale:
    # Rescale the image (get all values between 0 and 1)
    return img/255.
  else:
    return img