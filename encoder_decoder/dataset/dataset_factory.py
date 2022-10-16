from encoder_decoder.dataset.dataset import get_dataset


def create_dataset(number, data_number):
    if number == 0:
        return get_dataset(data_number)

    return None
