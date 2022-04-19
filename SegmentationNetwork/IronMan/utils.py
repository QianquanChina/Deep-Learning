import transforms as T

class SegmentationPresetTrain:# {{{

    def __init__(
                    self,
                    base_size, 
                    crop_size,
                    hflip_prob = 0.5, 
                    mean = ( 0.485, 0.456, 0.406 ), 
                    std  = ( 0.229, 0.224, 0.225 )
                ):

        min_size = int( 0.5 * base_size )
        max_size = int( 2.0 * base_size )
    
        trans = [ T.RandomResize( min_size, max_size ) ] 

        if hflip_prob > 0:

            trans.append( T.RandomHorizontalFlip(hflip_prob) ) #type: ignore

        trans.extend(
                        [   #type: ignore 
                            T.RandomCrop(crop_size),
                            T.ToTensor(),
                            T.Normalize( mean = mean, std = std ),
                        ]
                    )
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):

        return self.transforms( img, target ) # type: ignore}}}

class SegmentationPresetEval:# {{{

    def __init__(
                    self, 
                    base_size, 
                    mean = (0.485, 0.456, 0.406), 
                    std  = (0.229, 0.224, 0.225)

                ):

        self.transforms = T.Compose(
                                       [
                                           T.RandomResize(base_size, base_size), # type: ignore
                                           T.ToTensor(),
                                           T.Normalize(mean=mean, std=std),
                                       ]
                                   )

    def __call__(self, img, target):

        return self.transforms( img, target ) # type: ignore}}}

def get_transform(train):# {{{

    base_size = 520
    crop_size = 480

    return SegmentationPresetTrain( base_size, crop_size ) if train else SegmentationPresetEval(base_size)# }}}

