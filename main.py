import numpy as np
import cv2 as cv
import contextlib
from tensorflow import keras
from tensorflow.keras import layers

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
    channels = (True, True, True)
    cam_width, cam_height = 1920, 1080
    display_scale = 0.4
    display_width, display_height = (int(cam_width*display_scale),
                                     int(cam_height*display_scale))
    display_width -= display_width % 2
    person_display_width = display_width // 2

##    optimizer = keras.optimizers.Adam(learning_rate = 0.0005)
    
    # Input shape for the autoencoder, 3 channels (BGR)
    decompressed_shape = (display_height, person_display_width, 3)
    compressed_shape = (200,)

    encoder_model, encoder_input_layer, encoder_output_layer = encoder(decompressed_shape, *compressed_shape)
##    encoder_model.compile()
##    encoder_input_layer = encoder_model.get_layer(index=0)
##    encoder_output_layer = encoder_model.get_layer(index=-1)
    
    decoder_model_left, decoder_left_output_layer = decoder(compressed_shape, decompressed_shape, encoder_output_layer, name='decoder_left')
    decoder_model_right, decoder_right_output_layer = decoder(compressed_shape, decompressed_shape, encoder_output_layer, name='decoder_right')
##    decoder_model_left.compile()
##    decoder_model_right.compile()
##    decoder_left_output_layer = decoder_model_left.get_layer(index=-1)
##    decoder_right_output_layer = decoder_model_right.get_layer(index=-1)
    full_model_left = keras.Model(encoder_input_layer, decoder_left_output_layer, name='full_left_model')
    full_model_right = keras.Model(encoder_input_layer, decoder_right_output_layer, name='full_right_model')
    full_model_left.compile(optimizer='adam',
                            loss='binary_crossentropy',
                            metrics=['accuracy'])
    full_model_right.compile(optimizer='adam',
                            loss='binary_crossentropy',
                            metrics=['accuracy'])
    
##    decoder_left_output_layer = decoder_model_left.get_layer(index=-1)
##    decoder_right_output_layer = decoder_model_right.get_layer(index=-1)

##    trainable_model_left = keras.Model(encoder_input_layer, decoder_left_output_layer, name='Trainable Left')
##    trainable_model_right = keras.Model(encoder_input_layer, decoder_right_output_layer, name='Trainable Right')
    
##    sample_result = encoder_model(np.zeros((1, *decompressed_shape), dtype=np.float32))
##    sample_output = decoder_model_left(sample_result).numpy()
##    print('Sample output shape:', np.shape(sample_output[0]))
    
    with video_capture_wrapper(0, cv.CAP_DSHOW) as cap:
##        print(cap.getBackendName())
        cap.set(cv.CAP_PROP_FPS, 30.0)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, cam_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, cam_height)
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
##        print(cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT))
##        print(cap.get(cv.CAP_PROP_FPS))
##        print(string_ord(int(cap.get(cv.CAP_PROP_FOURCC))))
        i_did_the_thing = False
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            smol = cv.resize(frame, dsize=(display_width, display_height), interpolation=cv.INTER_CUBIC)
            
            person_left = smol[:,person_display_width:]
            person_right = smol[:,:person_display_width]
            
            person_left_flipped = cv.flip(person_left, 1)
            person_right_flipped = cv.flip(person_right, 1)
            
            people = (smol[:,display_width//2:], smol[:,:display_width//2])
            
            people = (np.reshape(person_left, (1, *decompressed_shape)),
                      np.reshape(person_right, (1, *decompressed_shape)))

            if not i_did_the_thing:
##                print(type(smol))
##                print(np.shape(people[0]), np.shape(people[1]))
                i_did_the_thing = True
            
            for i, has_channel in enumerate(channels):
                if not has_channel:
                    smol[:,:,i] = 0
            # Encode/decode here
            ##################################################################################
            
            person_left_training = np.reshape(person_left, (1, *decompressed_shape))
            person_right_training = np.reshape(person_right, (1, *decompressed_shape))

            full_model_left.fit(person_left_training, person_left_training)
            full_model_right.fit(person_right_training, person_right_training)
            
            fucked_person_left = full_model_right(person_left_training)[0]
            fucked_person_right = full_model_left(person_right_training)[0]

            
            top_row = np.concatenate((person_left_flipped, person_right_flipped), axis=1)
            # TODO swap the bottom row ##############
            bottom_row = np.concatenate((fucked_person_left, fucked_person_right), axis=1)
            vis = np.concatenate((top_row, bottom_row), axis=0)
    ##        cv.imshow('Webcam', smol)
            cv.imshow('Webcam', vis)
            
            key = chr(cv.waitKey(1) & 0xFF)
            if not ret or key == 'q':
                break
            elif key == 'b':
                channels = (not channels[0], channels[1],     channels[2])
            elif key == 'g':
                channels = (channels[0],     not channels[1], channels[2])
            elif key == 'r':
                channels = (channels[0],     channels[1],     not channels[2])
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
