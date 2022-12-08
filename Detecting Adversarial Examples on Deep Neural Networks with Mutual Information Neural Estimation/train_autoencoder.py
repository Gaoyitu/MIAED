import Autoencoders
import tensorflow as tf

def train_autoencoder(encoder,fc_model,decoder,x,batch_size,epochs,dataset_name,vector_dimension):

    autoencoder = Autoencoders.Autoencoder(encoder=encoder, fc_model=fc_model,decoder=decoder)

    autoencoder.compile(optimizer='adam',
                        loss='mse')

    autoencoder.fit(x, batch_size=batch_size, epochs=epochs)
    autoencoder.encoder.save_weights('networks/'+dataset_name+'/autoencoder/encoder_'+str(vector_dimension)+'.h5')
    autoencoder.fc_model.save_weights('networks/'+dataset_name+'/autoencoder/fc_model_'+str(vector_dimension)+'.h5')
    autoencoder.decoder.save_weights('networks/'+dataset_name+'/autoencoder/decoder_'+str(vector_dimension)+'.h5')

    return autoencoder


def train_MI_autoencoder(encoder,fc_model,decoder,discriminator_global,discriminator_local,discriminator_prior,x,conf,dataset_name,ablation):

    if(ablation == 'local'):
        conf.local_w = 0
        mi_autoencoder = Autoencoders.MI_Autoencoder(conf=conf,
                                                     encoder=encoder,
                                                     fc_model=fc_model,
                                                     decoder=decoder,
                                                     discriminator_global=discriminator_global,
                                                     discriminator_local=discriminator_local,
                                                     discriminator_prior=discriminator_prior)
        mi_autoencoder.train_MI_autoencoder(x,64)

        mi_autoencoder.encoder.save_weights('networks/' + dataset_name + '/autoencoder/mi_'+ablation+'_encoder.h5')
        mi_autoencoder.fc_model.save_weights('networks/' + dataset_name + '/autoencoder/mi_'+ablation+'_fc_model.h5')
        mi_autoencoder.decoder.save_weights('networks/' + dataset_name + '/autoencoder/mi_'+ablation+'_decoder.h5')

        return mi_autoencoder
    if(ablation == 'global'):
        conf.global_w = 0
        mi_autoencoder = Autoencoders.MI_Autoencoder(conf=conf,
                                                     encoder=encoder,
                                                     fc_model=fc_model,
                                                     decoder=decoder,
                                                     discriminator_global=discriminator_global,
                                                     discriminator_local=discriminator_local,
                                                     discriminator_prior=discriminator_prior)
        mi_autoencoder.train_MI_autoencoder(x,64)

        mi_autoencoder.encoder.save_weights('networks/' + dataset_name + '/autoencoder/mi_' + ablation + '_encoder.h5')
        mi_autoencoder.fc_model.save_weights(
            'networks/' + dataset_name + '/autoencoder/mi_' + ablation + '_fc_model.h5')
        mi_autoencoder.decoder.save_weights('networks/' + dataset_name + '/autoencoder/mi_' + ablation + '_decoder.h5')

        return mi_autoencoder
    if (ablation == 'prior'):
        conf.dw = 0
        conf.gpw = 0
        mi_autoencoder = Autoencoders.MI_Autoencoder(conf=conf,
                                                     encoder=encoder,
                                                     fc_model=fc_model,
                                                     decoder=decoder,
                                                     discriminator_global=discriminator_global,
                                                     discriminator_local=discriminator_local,
                                                     discriminator_prior=discriminator_prior)
        mi_autoencoder.train_MI_autoencoder(x,64)

        mi_autoencoder.encoder.save_weights('networks/' + dataset_name + '/autoencoder/mi_' + ablation + '_encoder.h5')
        mi_autoencoder.fc_model.save_weights(
            'networks/' + dataset_name + '/autoencoder/mi_' + ablation + '_fc_model.h5')
        mi_autoencoder.decoder.save_weights('networks/' + dataset_name + '/autoencoder/mi_' + ablation + '_decoder.h5')

        return mi_autoencoder
    if (ablation == 'local_global'):
        conf.local_w  = 0
        conf.global_w = 0
        mi_autoencoder = Autoencoders.MI_Autoencoder(conf=conf,
                                                     encoder=encoder,
                                                     fc_model=fc_model,
                                                     decoder=decoder,
                                                     discriminator_global=discriminator_global,
                                                     discriminator_local=discriminator_local,
                                                     discriminator_prior=discriminator_prior)
        mi_autoencoder.train_MI_autoencoder(x,64)

        mi_autoencoder.encoder.save_weights('networks/' + dataset_name + '/autoencoder/mi_' + ablation + '_encoder.h5')
        mi_autoencoder.fc_model.save_weights(
            'networks/' + dataset_name + '/autoencoder/mi_' + ablation + '_fc_model.h5')
        mi_autoencoder.decoder.save_weights('networks/' + dataset_name + '/autoencoder/mi_' + ablation + '_decoder.h5')

        return mi_autoencoder

    if (ablation == 'local_prior'):
        conf.local_w  = 0
        conf.dw = 0
        conf.gpw = 0
        mi_autoencoder = Autoencoders.MI_Autoencoder(conf=conf,
                                                     encoder=encoder,
                                                     fc_model=fc_model,
                                                     decoder=decoder,
                                                     discriminator_global=discriminator_global,
                                                     discriminator_local=discriminator_local,
                                                     discriminator_prior=discriminator_prior)
        mi_autoencoder.train_MI_autoencoder(x,64)

        mi_autoencoder.encoder.save_weights('networks/' + dataset_name + '/autoencoder/mi_' + ablation + '_encoder.h5')
        mi_autoencoder.fc_model.save_weights(
            'networks/' + dataset_name + '/autoencoder/mi_' + ablation + '_fc_model.h5')
        mi_autoencoder.decoder.save_weights('networks/' + dataset_name + '/autoencoder/mi_' + ablation + '_decoder.h5')

        return mi_autoencoder

    if (ablation == 'global_prior'):
        conf.global_w  = 0
        conf.dw = 0
        conf.gpw = 0
        mi_autoencoder = Autoencoders.MI_Autoencoder(conf=conf,
                                                     encoder=encoder,
                                                     fc_model=fc_model,
                                                     decoder=decoder,
                                                     discriminator_global=discriminator_global,
                                                     discriminator_local=discriminator_local,
                                                     discriminator_prior=discriminator_prior)
        mi_autoencoder.train_MI_autoencoder(x,64)

        mi_autoencoder.encoder.save_weights('networks/' + dataset_name + '/autoencoder/mi_' + ablation + '_encoder.h5')
        mi_autoencoder.fc_model.save_weights(
            'networks/' + dataset_name + '/autoencoder/mi_' + ablation + '_fc_model.h5')
        mi_autoencoder.decoder.save_weights('networks/' + dataset_name + '/autoencoder/mi_' + ablation + '_decoder.h5')

        return mi_autoencoder




def train_mnist_autoencoder():
    from models import get_mnist_encoder,get_mnist_full_connection_layers,get_mnist_decoder
    from data import get_mnist

    train_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=20,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5,
            patience=3, verbose=1
        )
    ]

    x_train,_,x_test,_ = get_mnist()

    autoencoder = train_autoencoder(encoder=get_mnist_encoder(),
                                    fc_model=get_mnist_full_connection_layers(),
                                    decoder=get_mnist_decoder(),
                                    x=x_train,
                                    batch_size=256,
                                    epochs=100,
                                    dataset_name='mnist',
                                    train_callbacks=train_callbacks)


def train_mnist_mi_autoencoder(ablation):
    from models import get_mnist_encoder, get_mnist_full_connection_layers, get_mnist_decoder,\
        get_discriminator_global,get_discriminator_local,get_discriminator_prior
    from data import get_mnist
    import config

    parser = config.get_arguments_mnist()

    conf = parser.parse_args()

    x_train, _, x_test, _ = get_mnist()

    mi_autoencoder = train_MI_autoencoder(encoder=get_mnist_encoder(),
                                           fc_model=get_mnist_full_connection_layers(64),
                                           decoder=get_mnist_decoder(64),
                                           discriminator_global=get_discriminator_global((7 * 7 * 32+64,)),
                                           discriminator_local=get_discriminator_local((7, 7, 32 + 64)),
                                           discriminator_prior=get_discriminator_prior((64,)),
                                           x=x_train,
                                           conf=conf,
                                           dataset_name='mnist',
                                          ablation=ablation)



def train_cifar10_autoencoder():
    from models import get_cifar10_encoder,get_cifar10_full_connection_layers,get_cifar10_decoder
    from data import get_cifar10

    train_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=20,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5,
            patience=3, verbose=1
        )
    ]

    x_train,_,x_test,_ = get_cifar10()

    autoencoder = train_autoencoder(encoder=get_cifar10_encoder(),
                                    fc_model=get_cifar10_full_connection_layers(64),
                                    decoder=get_cifar10_decoder(64),
                                    x=x_train,
                                    batch_size=256,
                                    epochs=100,
                                    dataset_name='cifar10',
                                    train_callbacks=train_callbacks)



def train_cifar10_mi_autoencoder(ablation):
    from models import get_cifar10_encoder, get_cifar10_full_connection_layers, get_cifar10_decoder,\
        get_discriminator_global,get_discriminator_local,get_discriminator_prior
    from data import get_cifar10
    import config

    parser = config.get_arguments_cifar10()

    conf = parser.parse_args()

    x_train, _, x_test, _ = get_cifar10()

    mi_autoencoder = train_MI_autoencoder(encoder=get_cifar10_encoder(),
                                           fc_model=get_cifar10_full_connection_layers(64),
                                           decoder=get_cifar10_decoder(64),
                                           discriminator_global=get_discriminator_global((8 * 8 * 64+64,)),
                                           discriminator_local=get_discriminator_local((8, 8, 64 + 64)),
                                           discriminator_prior=get_discriminator_prior((64,)),
                                           x=x_train,
                                           conf=conf,
                                           dataset_name='cifar10',
                                          ablation=ablation)


def train_imagenet_mi_autoencoder(ablation):
    from models import get_imagenet_encoder, get_imagenet_full_connection_layers, get_imagenet_decoder,\
        get_discriminator_global,get_discriminator_local,get_discriminator_prior
    from data import get_imagenet
    import config

    parser = config.get_arguments_imagenet()

    conf = parser.parse_args()

    x_train, _, _, _ = get_imagenet()

    mi_autoencoder = train_MI_autoencoder(encoder=get_imagenet_encoder(),
                                           fc_model=get_imagenet_full_connection_layers(64),
                                           decoder=get_imagenet_decoder(64),
                                           discriminator_global=get_discriminator_global((14 * 14 * 128+64,)),
                                           discriminator_local=get_discriminator_local((14, 14, 128 + 64)),
                                           discriminator_prior=get_discriminator_prior((64,)),
                                           x=x_train,
                                           conf=conf,
                                           dataset_name='imagenet',
                                          ablation=ablation)


if __name__ == '__main__':
    ablations = ['local','global','prior','local_global','local_prior','global_prior']
    for ablation in ablations:
        print(ablation)
        train_imagenet_mi_autoencoder(ablation=ablation)








