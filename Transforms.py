from torchvision import transforms

use_resnet = True
image_size = (48, 48) if use_resnet else (32, 32)

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])

transformations = {

    # Resize, normalize and jitter image brightness -----------------
    'data_jitter_brightness': transforms.Compose([
        transforms.Resize(image_size),
        transforms.ColorJitter(brightness=5),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ]),

    # Resize, normalize and jitter image saturation -----------------
    'data_jitter_saturation': transforms.Compose([
        transforms.Resize(image_size),
        transforms.ColorJitter(saturation=5),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ]),

    # Resize, normalize and jitter image contrast
    'data_jitter_contrast': transforms.Compose([
        transforms.Resize(image_size),
        transforms.ColorJitter(contrast=5),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ]),

    # Resize, normalize and jitter image hues -----------------
    'data_jitter_hue': transforms.Compose([
        transforms.Resize(image_size),
        transforms.ColorJitter(hue=0.4),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ]),

    # Resize, normalize and rotate image
    'data_rotate': transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ]),

    # Resize, normalize and flip image horizontally and vertically
    'data_hvflip': transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(1),
        transforms.RandomVerticalFlip(1),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ]),

    # Resize, normalize and flip image horizontally
    'data_hflip': transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(1),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ]),

    # Resize, normalize and flip image vertically
    'data_vflip': transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomVerticalFlip(1),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ]),

    # Resize, normalize and shear image
    'data_shear': transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomAffine(degrees=15, shear=2),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ]),

    # Resize, normalize and translate image
    'data_translate': transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ]),

    """# Resize, normalize and crop image -----------------
    'data_center': transforms.Compose([
        transforms.Resize((55, 55)),
        transforms.CenterCrop(48),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ]),"""

    # Resize, normalize and convert image to grayscale -----------------
    'data_grayscale': transforms.Compose([
        transforms.Resize(image_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])

}
