import argparse

def get_arguments_mnist():
    parser = argparse.ArgumentParser()

    parser.add_argument('--imageH', type=int, default=28, help='the hight of image, default = 28')
    parser.add_argument('--imageW', type=int, default=28, help='the width of image, default = 28')
    parser.add_argument('--channel', type=int, default=1, help='the channel of image, default=1')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs, default = 100')

    parser.add_argument('--rec_w', type=float, default= 10.0, help='the reconstruction weight, default=10.')
    parser.add_argument('--global_w',type=float, default=0.1, help='the weight of global MI, default=0.001')
    parser.add_argument('--local_w', type=float, default=0.1, help='the weight of local MI, default=0.001')
    parser.add_argument('--dw', type=float, default=0.0001, help='the weight of forward distribution, default=0.0001')
    parser.add_argument('--gpw', type=float, default=10., help='the weight of gradient penalty. default=10.')

    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate, default=0.0001')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

    parser.add_argument('--batch_size', type=int, default=256, help='batch size, default=512')


    return parser


def get_arguments_cifar10():
    parser = argparse.ArgumentParser()

    parser.add_argument('--imageH', type=int, default=32, help='the hight of image, default = 32')
    parser.add_argument('--imageW', type=int, default=32, help='the width of image, default = 32')
    parser.add_argument('--channel', type=int, default=3, help='the channel of image, default=3')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs, default = 100')

    parser.add_argument('--rec_w', type=float, default=10.0, help='the reconstruction weight, default=10.')
    parser.add_argument('--global_w', type=float, default=0.001, help='the weight of global MI, default=0.001')
    parser.add_argument('--local_w', type=float, default=0.001, help='the weight of local MI, default=0.001')
    parser.add_argument('--dw', type=float, default=0.0001, help='the weight of forward distribution, default=0.0001')
    parser.add_argument('--gpw', type=float, default=10., help='the weight of gradient penalty. default=10.')

    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate, default=0.0001')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

    parser.add_argument('--batch_size', type=int, default=256, help='batch size, default=256')


    return parser



def get_arguments_imagenet():
    parser = argparse.ArgumentParser()

    parser.add_argument('--imageH', type=int, default=224, help='the hight of image, default = 28')
    parser.add_argument('--imageW', type=int, default=224, help='the width of image, default = 28')
    parser.add_argument('--channel', type=int, default=3, help='the channel of image, default=1')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs, default = 20')

    parser.add_argument('--rec_w', type=float, default=10.0, help='the reconstruction weight, default=10.0')
    parser.add_argument('--global_w', type=float, default=0.01, help='the weight of global MI, default=0.01')
    parser.add_argument('--local_w', type=float, default=0.01, help='the weight of local MI, default=0.01')
    parser.add_argument('--dw', type=float, default=0.001, help='the weight of forward distribution, default=0.001')
    parser.add_argument('--gpw', type=float, default=10., help='the weight of gradient penalty. default=10.')

    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate, default=0.0001')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

    parser.add_argument('--batch_size', type=int, default=16, help='batch size, default=16')


    return parser