import numpy as np
import cv2 as cv
import contextlib
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K

@contextlib.contextmanager
def video_capture_wrapper(*args, **kwargs):
    cap = cv.VideoCapture(*args, **kwargs)
    if not cap.isOpened():
        print('opening failed')
        return
    try:
        yield cap
    finally:
        cap.release()

def string_ord(num):
    ret_string = ""
    while num != 0:
        ret_string += chr(num & 0xFF)
        num //= 0x100
    return ret_string

def encoder(input_encoder, compression_length):
    inputs = keras.Input(shape=input_encoder, name='input_layer')
    # Block 1
    x = layers.Conv2D(32, kernel_size=3, strides = 1, padding='same', name='conv_1')(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.LeakyReLU(name='lrelu_1')(x)

    # Block 2
    # Setting stride to 2 means check every other point. So half the size.
    x = layers.Conv2D(64, 3, 2, padding='same', name='conv_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.LeakyReLU(name='lrelu_2')(x)

    # Block 3
    # Half again
    x = layers.Conv2D(64, 3, 2, padding='same', name='conv_3')(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.LeakyReLU(name='lrelu_3')(x)
    
    # Block 4
    x = layers.Conv2D(64, 3, 1, padding='same', name='conv_4')(x)
    x = layers.BatchNormalization(name='bn_4')(x)
    x = layers.LeakyReLU(name='lrelu_4')(x)
    
    # Final Block
    flatten = layers.Flatten()(x)
    bottleneck = layers.Dense(compression_length, name='dense_1')(flatten)
    
    model = keras.Model(inputs, bottleneck, name='encoder')
    return model, inputs, bottleneck

def decoder(input_decoder, expanded_shape, inputs=None, name='decoder'):
    h, w, channels = expanded_shape
    h, w = h // 4, w //4
    print(h, w)
    # Initial Block
    if inputs is None:
        inputs = keras.Input(shape=input_decoder, name='input_layer')
    x = layers.Dense(h*w*64, name='dense_2')(inputs)
    x = layers.Reshape((h, w, 64), name='Reshape_Layer')(x)
    
    # Block 1
    x = layers.Conv2DTranspose(64, 3, strides= 1, padding='same',name='conv_transpose_1')(x)
    x = layers.BatchNormalization(name='bn_5')(x)
    x = layers.LeakyReLU(name='lrelu_5')(x)
    
    # Block 2
    x = layers.Conv2DTranspose(64, 3, strides= 2, padding='same', name='conv_transpose_2')(x)
    x = layers.BatchNormalization(name='bn_6')(x)
    x = layers.LeakyReLU(name='lrelu_6')(x)
   
    # Block 3
    x = layers.Conv2DTranspose(32, 3, 2, padding='same', name='conv_transpose_3')(x)
    x = layers.BatchNormalization(name='bn_7')(x)
    x = layers.LeakyReLU(name='lrelu_7')(x)
   
    # Block 4
    # Channels is probably 3 for BGR but in case I change it later...
    outputs = layers.Conv2DTranspose(channels, 3, 1, padding='same', activation='sigmoid', name='conv_transpose_4')(x)
    model = keras.Model(inputs, outputs, name=name)
    return model, outputs

def main():
    batch_size = 200
    starting_learning_rate = 0.01
    later_learning_rate = 0.001
    
    channels = (True, True, True)
    cam_width, cam_height = 1920, 1080
    display_scale = 0.4
    display_width, display_height = (int(cam_width*display_scale),
                                     int(cam_height*display_scale)*2)
    display_width -= display_width % 2
    person_display_width = display_width // 2
    person_display_height = display_height // 2
    print(display_width, display_height)
    
    # Input shape for the autoencoder, 3 channels (BGR)
    decompressed_shape = (person_display_height, person_display_width, 3)
    print('decompressed_shape: ', decompressed_shape)
    compressed_shape = (200,)

    encoder_model, encoder_input_layer, encoder_output_layer = encoder(decompressed_shape, *compressed_shape)

    decoder_model_left, decoder_left_output_layer = decoder(compressed_shape, decompressed_shape, encoder_output_layer, name='decoder_left')
    decoder_model_right, decoder_right_output_layer = decoder(compressed_shape, decompressed_shape, encoder_output_layer, name='decoder_right')

    full_model_left = keras.Model(encoder_input_layer, decoder_left_output_layer, name='full_left_model')
    full_model_right = keras.Model(encoder_input_layer, decoder_right_output_layer, name='full_right_model')

    optimizer = keras.optimizers.Adam(learning_rate = starting_learning_rate)
    full_model_left.compile(optimizer=optimizer,
                            loss='binary_crossentropy',
                            metrics=['accuracy'])
    full_model_right.compile(optimizer=optimizer,
                            loss='binary_crossentropy',
                            metrics=['accuracy'])

    # Index 0 is the number of the camera.
    # TODO dynamic camera choice
    with video_capture_wrapper(0, cv.CAP_DSHOW) as cap:
        cap.set(cv.CAP_PROP_FPS, 30.0)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, cam_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, cam_height)
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
        
        i_did_the_thing = False
        
        count = 0
        frames_index = 0
        
        frames_left = np.zeros((batch_size, *decompressed_shape), np.float32)
        frames_right = np.zeros((batch_size, *decompressed_shape), np.float32)
        
        bottom_row = np.zeros((person_display_height, display_width, 3), np.uint8)
        
        while cap.isOpened():
            ret, frame = cap.read()
            smol = cv.resize(frame, dsize=(display_width, person_display_height), interpolation=cv.INTER_CUBIC)
            
            person_left = smol[:,person_display_width:]
            person_right = smol[:,:person_display_width]
            
            person_left_flipped = cv.flip(person_left, 1)
            person_right_flipped = cv.flip(person_right, 1)

            person_left_training = np.reshape(person_left/255, (1, *decompressed_shape))
            person_right_training = np.reshape(person_right/255, (1, *decompressed_shape))

            if frames_index == batch_size:
                frames_index = 0
                print('Training...')

                # TODO train models simultaneously
                # https://stackoverflow.com/a/44873889/9081715
                full_model_left.fit(frames_left, frames_left)
                full_model_right.fit(frames_right, frames_right)

                if not i_did_the_thing:
                    K.set_value(full_model_left.optimizer.learning_rate, later_learning_rate)
                    K.set_value(full_model_right.optimizer.learning_rate, later_learning_rate)
                    i_did_the_thing = True
                
##                frames_left = np.zeros((batch_size, *decompressed_shape), np.float32)
##                frames_right = np.zeros((batch_size, *decompressed_shape), np.float32)
            
            count += 1
            if count == 1:
                count = 0
                print(f'new training frame [{frames_index+1}/{batch_size}]')
                frames_left[frames_index] = person_left/255
                frames_right[frames_index] = person_right/255
                frames_index += 1
                
##                fucked_person_left = (full_model_right(person_left_training).numpy()[0]*255).astype('uint8')
##                fucked_person_right = (full_model_left(person_right_training).numpy()[0]*255).astype('uint8')
                
                fucked_person_left = cv.flip((full_model_left(person_left_training).numpy()[0]*255).astype('uint8'), 1)
                fucked_person_right = cv.flip((full_model_right(person_right_training).numpy()[0]*255).astype('uint8'), 1)
                
                bottom_row = np.concatenate((fucked_person_left, fucked_person_right), axis=1)

            top_row = np.concatenate((person_left_flipped, person_right_flipped), axis=1)
##            bottom_row = np.concatenate((person_right_flipped, person_left_flipped), axis=1)
            vis = np.concatenate((top_row, bottom_row), axis=0)
##            cv.imshow('Webcam', top_row)
            cv.imshow('Webcam', vis)
            
            key = chr(cv.waitKey(1) & 0xFF)
            if not ret or key == 'q':
                break

    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
