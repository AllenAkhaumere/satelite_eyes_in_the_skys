import torchvision

def sat_augmentation():
    
    """ EuroSAT data augmentation"""
    mean, std_dev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    training_transforms = transforms.Compose([transforms.Resize((100, 100)),
                                              transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(100),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomGrayscale(p=0.1),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=mean, std=std_dev)])

    validation_transforms = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std_dev)])

    return training_transforms, validation_transforms