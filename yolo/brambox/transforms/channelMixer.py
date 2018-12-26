#
#   Copyright EAVISE
#   By Tanguy Ophoff
#

import logging as log
#log = logging.getLogger(__name__)   # noqa

from PIL import Image
import numpy as np
try:
    import cv2
except ModuleNotFoundError:
    log.debug('OpenCV not installed, always using PIL')
    cv2 = None

__all__ = ['ChannelMixer']


class ChannelMixer:
    """ Mix channels of multiple inputs in a single output image.
    This class works with opencv_ images (np.ndarray), and will mix the channels of multiple images into one new image.

    Args:
        num_channels (int, optional): The number of channels the output image will have; Default **3**

    Example:
        >>> # Replace the 3th channel of an image with a channel from another image
        >>> mixer = brambox.transforms.ChannelMixer()
        >>> mixer.set_channels([(0,0), (0,1), (1,0)])
        >>> out = mixer(img1, img2)
        >>> # out => opencv image with channels: [img0_channel0, img0_channel1, img1_channel0]
    """
    def __init__(self, num_channels=3):
        self.num_channels = num_channels
        self.channels = [(0, i) for i in range(num_channels)]

    def set_channels(self, channels):
        """ Set from which channels the output image should be created.
        The channels list should have the same length as the number of output channels.

        Args:
            channels (list): List of tuples containing (img_number, channel_number)
        """
        if len(channels) != self.num_channels:
            raise ValueError('You should have one [image,channel] per output channel')
        self.channels = [(c[0], c[1]) for c in channels]

    def __call__(self, *imgs):
        """ Create and return output image.

        Args:
            *imgs: Argument list with all the images needed for the mix

        Warning:
            Make sure the images all have the same width and height before mixing them.
        """
        m = max(self.channels, key=lambda c: c[0])[0]
        if m >= len(imgs):
            raise ValueError(f'{m} images are needed to perform the mix')

        if isinstance(imgs[0], Image.Image):
            pil_image = True
            imgs = [np.array(img) for img in imgs]
        else:
            pil_image = False

        res = np.zeros([imgs[0].shape[0], imgs[0].shape[1], self.num_channels], 'uint8')
        for i in range(self.num_channels):
            if imgs[self.channels[i][0]].ndim >= 3:
                res[..., i] = imgs[self.channels[i][0]][..., self.channels[i][1]]
            else:
                res[..., i] = imgs[self.channels[i][0]]
        res = np.squeeze(res)

        if pil_image:
            return Image.fromarray(res)
        else:
            return res
