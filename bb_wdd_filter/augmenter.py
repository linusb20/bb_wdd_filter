from typing import List, Dict
import albumentations as A
import numpy as np

class AugmenterAlbumentations:
    def __init__(self) -> None:
        pass

    def augment_video(video:List, augmenters:List):
        '''
            INPUT:
                video       list of images
                augmenters  list of albumentations videos


            OUTPUT:         list of images (augmented in the same way)

        '''
        # AUGMENTER INIT
        dictoINIT =  {} 
        dictoCALL = {}
        for i in range(1,len(video)):
            dictoINIT[f'image{i}'] = "image"
            dictoCALL[f'image{i}'] = video[i]
    
        # same AUGMENTER for entire video (same transformations) 
        aug_compose =  A.Compose(
            augmenters,
            additional_targets=dictoINIT
        )

        # apply augmenter
        aug_video_dict = aug_compose(image=video[0], **dictoCALL)

        # transform dict to list
        aug_video = [
                aug_img.astype(np.float32) 
                for aug_img 
                in aug_video_dict.values()
            ]


        return aug_video