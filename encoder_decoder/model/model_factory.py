from encoder_decoder.model.unet import get_model as unet_model
from encoder_decoder.model.unet_low import get_model as unet_low_model
from encoder_decoder.model.unet_vector import get_model as unet_vector_model


def create_model(name, input_size=(512, 512, 3)):
    model = None
    if name == 'unet':
        model = unet_model(input_size=input_size)

    if name == 'unet_low':
        model = unet_low_model(input_size=input_size)

    if name == 'unet_vector':
        model = unet_vector_model(input_size=input_size)

    print(model.summary())
    return model
