import random
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Dropout, Flatten, Lambda
from keras.layers.advanced_activations import PReLU
from keras.utils.visualize_util import plot
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

# paths
PATH = './data/'
VAL_PATH = './data/'
# training hyperparameters
EPOCHS = 20
BATCH_SIZE = 256
# DROPOUT = .40  # car is a little jerky
DROPOUT = .34  # smooth drive
#DROPOUT = .20  # car weaves too much
# file names
FILE = 'driving_log.csv'


class Preprocess:
    """
    Static class containing dataset preprocessing functions and model training constants.
    """

    @staticmethod
    def truncate_highly_logged_angles():
        """
        Create angle telemetry lists sized 0.1, then reduce the large lists,
        to make more even distribution of examples and reduce bias.
        """
        global log

        log = pd.read_csv(PATH + FILE)

        # Break the log into separate lists for each set of angles from -1 to 1 with a step size of 0.1.
        angles_lists = []
        for i in np.arange(-1.0, 1.0, 0.1):  #
            angles_lists.append(log[(log['steering'] > i) & (log['steering'] < (i + 0.1))])

        truncated_log_list = angles_lists[0]

        # Truncate angles counted at 2584 down to 610 items.
        for i in range(1, len(angles_lists)):
            if angles_lists[i].size < 2584:
                truncated_log_list = pd.concat([truncated_log_list, angles_lists[i]])
            else:
                truncated_log_list = pd.concat([truncated_log_list, angles_lists[i].iloc[0:610, :]])

        log = truncated_log_list

    @staticmethod
    def create_left_right_steering_angles():
        """
        Telemetry is provided based on centre camera images.
        This function takes centre image angle then
        adds 0.25 to create left image angles,
        subtracts 0.25 to create right image angles and
        pandas to concatenate into X images dataset and new y labels dataset.
        """
        global X_images, y_labels

        X_images = pd.concat([log.ix[:, 0], log.ix[:, 1], log.ix[:, 2]])
        y_labels = pd.concat([log.ix[:, 3], log.ix[:, 3] + 0.25, log.ix[:, 3] - 0.25])  # 0.
        # y_labels = pd.concat([log.ix[:, 3], log.ix[:, 3] + 0.3, log.ix[:, 3] - 0.3])  # 1. smooth but crashes track 2 top mountain tight turn
        # y_labels = pd.concat([log.ix[:, 3], log.ix[:, 3] + 0.4, log.ix[:, 3] - 0.4])  # 2. weaves more, crashes similar location as 1. above

    @staticmethod
    def split_out_training_and_validation_datasets():
        """
         Split 80% / 20% telemetry dataset for Training / Validation use.
        """
        global X_train, X_test, y_train, y_test, num_train_images, num_test_images
        # Break the driving log into Training(80%) and Testing(20%) set.
        X_train, X_test, y_train, y_test = train_test_split(X_images, y_labels, test_size=0.20, random_state=42)
        num_train_images = len(X_train)
        num_test_images = len(X_test)

    @staticmethod
    def load_image(filename, log_path):
        """
        Uses cv2.COLOUR_BFR2RGB.

        :param filename: Name of the file to load
        :param log_path: Name of the directory
        :return: return an image in RGB color Space
        """
        return cv2.cvtColor(cv2.imread(log_path + str.strip(filename)), cv2.COLOR_BGR2RGB)

    @staticmethod
    def random_shadow(img):
        """
        Randomly generates a rectangle from top to bottom of the image aka the mask.
        The pixel intensity within this mask image is randomly reduced within the range aka the random shadow effect.
        The mask is then merged with the original image to apply the random shadow.

        :param img: image to shadow
        :return: randomly shadowed image
        """
        # Randomise starting point.
        random.seed()
        # Randomly generate a rectangle
        rect = [[random.randint(0, 100), 0], [random.randint(100, 200), 0], [random.randint(100, 200), 66],
                [random.randint(0, 100), 66]]
        # Create the numpy array polygon
        poly = np.array([rect], dtype=np.int32)
        # Create the zero intialised mask.
        imgroi = np.zeros((66, 200), np.uint8)
        # Create the randomly shadowed mask.
        cv2.fillPoly(imgroi, poly, np.random.randint(50, 100))
        # Create a 3 data channel shadow mask.
        img3 = cv2.merge([imgroi, imgroi, imgroi])
        # Apply shadow mask to original image by subtracting pixel intensities.
        dst = cv2.subtract(img, img3)
        return dst

    @staticmethod
    def randomise_image_brightness(image):
        """
        Brightness - referenced Vivek Yadav post
        https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.yh93soib0
        :param image: Input image
        :return: return an image in RGB Color space with randomly modified image brightness.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        bv = .4 + np.random.uniform()
        hsv[::2] = hsv[::2] * bv

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    @staticmethod
    def transform_image_and_steering_angle(image, steering):
        """
        Horizontal shift of images (x-axis) to simulate car being at different positions on the road.
        Adjust steering angle by [-0.008 ... 0.008] per pixel shift.
        Vertical shift of image (y-axis) to simulate effect of car driving up or down sloped road.

        # Based on Vivek Yadav post
        # https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.yh93soib0
        :param image : input image
        :param steering : steering angle
        :return: return an image which is randomly shifted and the corresponding steering angle adjustment.
        """

        rows, cols, _ = image.shape
        trans_range = 100
        number_of_pixels = 10
        val_pixels = 0.4
        # trans_x = [-50.0 ... 50.0]
        trans_x = trans_range * np.random.uniform() - trans_range / 2
        # steering + ([-0.625 ... 0.625] / 80) => steering + [-0.008 ... 0.008]
        steering_angle = steering + trans_x / trans_range * 2 * val_pixels  # (trans_range * 2 * val_pixels) = 80
        # trans_y = [-5.0 ... 5.0]
        trans_y = number_of_pixels * np.random.uniform() - number_of_pixels / 2
        trans_mat = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        image = cv2.warpAffine(image, trans_mat, (cols, rows))
        return image, steering_angle

    @staticmethod
    def crop_camera_image(img, crop_height=66, crop_width=200):
        """
        Crop camera image to fit nvidia model input shape
        :param img: Input image data
        :param crop_height: cropped height required
        :param crop_width: returned width required
        :return: return an image with the desired height and width
        """
        height = img.shape[0]
        width = img.shape[1]
        y_start = 60
        x_start = int(width / 2) - int(crop_width / 2)
        return img[y_start:y_start + crop_height, x_start:x_start + crop_width]

    @staticmethod
    def training_data_generator(path=PATH, batch_size=128):
        """
        Python Generator that will Preprocess and Augment new images, one at a time in each batch.

        Preprocess :
            1. crop image to 66 x 200 trapezoid
            2. randomly shadow
        Augment :
            3. shift image and adjust steering angle
            4. randomly flip 50% of images and their angles

        :param path: Path to the data only pass the parent folder
        :param batch_size: Batch size required to be returned by this generator
        :return: return a list of images and steering angles
        """
        training_batch_pointer = 0
        while True:
            features = []
            labels = []
            for i in range(0, batch_size):
                row = (training_batch_pointer + i) % num_train_images
                # crop image to 66 x 200 trapezoid
                image = Preprocess.crop_camera_image(Preprocess.load_image(X_train.iloc[row], path))
                steering = y_train.iloc[row]
                # randomly shadow
                image = Preprocess.random_shadow(image)
                # shift image and adjust steering angle
                image, steering = Preprocess.transform_image_and_steering_angle(image, steering)
                # randomly flip 50% of images and their angles
                if random.random() >= .5:
                    image = cv2.flip(image, 1)
                    steering = -steering

                features.append(image)
                labels.append(steering)

            training_batch_pointer += batch_size

            yield (np.array(features), np.array(labels))

    @staticmethod
    def validation_data_generator(path=VAL_PATH, batch_size=128):
        """
        Python Generator that will Preprocess images, one at a time in each batch.

        Preprocess :
            1. crop image to 66 x 200 trapezoid

        :param path: Path to the data only pass the parent folder
        :param batch_size: Batch size required to be returned by this generator
        :return: return a list of images and steering angles
        """
        validation_batch_pointer = 0
        while True:
            features = []
            labels = []

            for i in range(0, batch_size):
                row = (validation_batch_pointer + i) % num_test_images
                # crop
                features.append(Preprocess.crop_camera_image(Preprocess.load_image(X_test.iloc[row], path)))
                labels.append(y_test.iloc[row])

            validation_batch_pointer += batch_size
            yield (np.array(features), np.array(labels))

    @staticmethod
    def build_nvidia_model(img_height=66, img_width=200, img_channels=3, dropout=DROPOUT):
        """
        Based on Nvidia network from "End to End Learning for Self-Driving Cars"
        http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

        :return: Sequential neural network model
            with lambda smoothing,
            PReLU activations for each layer,
            3 x Fully Connected layers with Dropouts to avoid overfitting,
            nadam optimisation
        """
        # build sequential model
        model = Sequential()

        # normalisation layer
        img_shape = (img_height, img_width, img_channels)
        model.add(
            Lambda(lambda x: (x - 127.5) / 255, input_shape=(img_shape), output_shape=(img_shape),
                   name='Normalization'))

        model.add(Convolution2D(32, 5, 5, border_mode='valid', subsample=(2, 2)))  # , activation='elu'))
        model.add(PReLU())
        model.add(Convolution2D(64, 5, 5, border_mode='valid', subsample=(2, 2)))  # , activation='prelu'))
        model.add(PReLU())
        model.add(Convolution2D(128, 5, 5, border_mode='valid', subsample=(2, 2)))  # , activation='prelu'))
        model.add(PReLU())
        # model.add(Dropout(dropout))

        model.add(Convolution2D(128, 3, 3, border_mode='valid', subsample=(1, 1)))  # , activation='prelu'))
        model.add(PReLU())
        model.add(Convolution2D(128, 3, 3, border_mode='valid', subsample=(1, 1)))  # , activation='prelu'))
        model.add(PReLU())
        model.add(Dropout(dropout))

        # flatten layer
        model.add(Flatten())

        # fully connected layers with dropout
        #model.add(Dense(512))  # ,activation='prelu'))  # 1. smoothest, crashes before middle of Test track 2
        #model.add(PReLU())
        #model.add(Dropout(dropout))

        model.add(Dense(256))  # ,activation='prelu'))  # 2. less smooth, crashes later, tight turn mountain top
        model.add(PReLU())
        model.add(Dropout(dropout))

        model.add(Dense(128))  # ,activation='prelu'))  # 3. litte jerky, crashes later than location of 2. above
        #model.add(Dense(100))  # ,activation='prelu'))  # 4. little jerky and crashes same location as 2. above
        model.add(PReLU())
        model.add(Dropout(dropout))

        model.add(Dense(64))  # ,activation='prelu'))  # 3. little jerky and crashes just a bit further than location of 2. above
        #model.add(Dense(50))  # ,activation='prelu'))
        model.add(PReLU())
        model.add(Dropout(dropout))

        model.add(Dense(1, name='OutputAngle'))  # ,  activation='relu', name='Out'\
        model.compile(optimizer='nadam', loss='mse')
        return model


# Start of the main function
if __name__ == '__main__':
    # Edit these for your own environment.
    # Preprocess.set_parameters()

    # Even out angle telemetry distribution.
    Preprocess.truncate_highly_logged_angles()

    # Generate more simulated telemetry angle data.
    Preprocess.create_left_right_steering_angles()

    # Separate datasets, ready for model training / validation.
    Preprocess.split_out_training_and_validation_datasets()

    # build model and display layers
    model = Preprocess.build_nvidia_model(dropout=DROPOUT)
    print(model.summary())

    plot(model, to_file='model.png', show_shapes=True)

    checkpoint = ModelCheckpoint("checkpoints/model-{val_loss:.4f}.h5",
                                 monitor='val_loss', verbose=1,
                                 save_weights_only=True, save_best_only=True)

    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)

    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')

    model.fit_generator(Preprocess.training_data_generator(PATH, BATCH_SIZE),
                        samples_per_epoch=BATCH_SIZE * int(num_train_images / BATCH_SIZE),
                        nb_epoch=EPOCHS, callbacks=[earlystopping, checkpoint],
                        validation_data=Preprocess.validation_data_generator(VAL_PATH, BATCH_SIZE),
                        nb_val_samples=num_test_images)

    # save weights and model
    # model.save_weights('model.h5')
    model.save_weights('model_weights.h5')
    # print(json.dumps(model.to_json(), sort_keys=True, indent=4))
    with open('model.json', 'w') as modelfile:
        # json.dump(model.to_json(), modelfile)
        modelfile.write(model.to_json())

    # Fri, 10/Feb/2017 new project submission requirement : combine weights (.h5) + model (.json) into model.h5.
    model.save('model.h5')
