
from cleverhans_l.future.tf2.attacks import fast_gradient_method,projected_gradient_descent,momentum_iterative_method
import numpy as np
import tensorflow as tf

import models



def generate_adv_examples_PGD(model,x,y,batch_size,eps,eps_iter):

    index = 0

    iters = x.shape[0]//batch_size
    print(iters)

    x_batch = x[index:index + batch_size]
    y_batch = y[index:index + batch_size]
    index = index + batch_size
    print(x_batch.shape)

    adv = projected_gradient_descent(model_fn=model,
                                     x=x_batch, eps=eps,
                                     eps_iter=eps_iter,
                                     nb_iter=50,
                                     norm=np.inf,
                                     clip_min=0.0,
                                     clip_max=1.0,
                                     y=y_batch,
                                     targeted=False,
                                     rand_init=True,
                                     sanity_checks=False)
    output = adv

    for i in range(1,iters,1):
        x_batch = x[index:index+batch_size]
        y_batch = y[index:index+batch_size]
        index = index + batch_size
        adv = projected_gradient_descent(model_fn=model,
                                         x=x_batch, eps=eps,
                                         eps_iter=eps_iter,
                                         nb_iter=50,
                                         norm=np.inf,
                                         clip_min=0.0,
                                         clip_max=1.0,
                                         y=y_batch,
                                         targeted=False,
                                         rand_init=True,
                                         sanity_checks=False)
        print(adv.numpy().shape)
        output = tf.concat([output,adv],0)
    print(output.numpy().shape)

    return output
def generate_adv_examples_BIM(model,x,y,batch_size,eps,eps_iter):

    index = 0

    iters = x.shape[0]//batch_size
    print(iters)

    x_batch = x[index:index + batch_size]
    y_batch = y[index:index + batch_size]
    index = index + batch_size
    print(x_batch.shape)

    adv = projected_gradient_descent(model_fn=model,
                                     x=x_batch, eps=eps,
                                     eps_iter=eps_iter,
                                     nb_iter=50,
                                     norm=np.inf,
                                     clip_min=0.0,
                                     clip_max=1.0,
                                     y=y_batch,
                                     targeted=False,
                                     rand_init=False,
                                     sanity_checks=False)
    output = adv

    for i in range(1,iters,1):
        x_batch = x[index:index+batch_size]
        y_batch = y[index:index+batch_size]
        index = index + batch_size
        adv = projected_gradient_descent(model_fn=model,
                                         x=x_batch, eps=eps,
                                         eps_iter=eps_iter,
                                         nb_iter=50,
                                         norm=np.inf,
                                         clip_min=0.0,
                                         clip_max=1.0,
                                         y=y_batch,
                                         targeted=False,
                                         rand_init=False,
                                         sanity_checks=False)
        print(adv.numpy().shape)
        output = tf.concat([output,adv],0)
    print(output.numpy().shape)

    return output
def generate_adv_examples_SPSA(model,x,y,batch_size,eps):

    index = 0

    iters = x.shape[0]//batch_size
    print(iters)

    x_batch = x[index:index + batch_size]
    y_batch = y[index:index + batch_size]
    index = index + batch_size
    print(x_batch.shape)

    adv = spsa(model_fn=model,
                          x=x_batch,
                          y=y_batch,
                          eps=eps,
                          nb_iter=20,
                          clip_min=0.0,
                          clip_max=1.0,
                          targeted=False)
    output = adv
    for i in range(1, iters, 1):
       print(i)
       x_batch = x[index:index + batch_size]
       y_batch = y[index:index + batch_size]
       index = index + batch_size
       adv = spsa(model_fn=model,
                  x=x_batch,
                  y=y_batch,
                  eps=eps,
                  nb_iter=20,
                  clip_min=0.0,
                  clip_max=1.0,
                  targeted=False)
       print(adv.numpy().shape)
       output = tf.concat([output, adv], 0)

    print(output.numpy().shape)

    return output

def generate_adv_examples_MIM(model,x,y,batch_size,eps,eps_iter):

    index = 0

    iters = x.shape[0]//batch_size
    print(iters)

    x_batch = x[index:index + batch_size]
    y_batch = y[index:index + batch_size]
    index = index + batch_size
    print(x_batch.shape)

    adv = momentum_iterative_method(model_fn=model,
                                     x=x_batch, eps=eps,
                                     eps_iter=eps_iter,
                                     nb_iter=50,
                                     norm=np.inf,
                                     clip_min=0.0,
                                     clip_max=1.0,
                                     y=y_batch,
                                     targeted=False,
                                     decay_factor=1.0,
                                     sanity_checks=False)
    output = adv

    for i in range(1,iters,1):
        x_batch = x[index:index+batch_size]
        y_batch = y[index:index+batch_size]
        index = index + batch_size
        adv = momentum_iterative_method(model_fn=model,
                                        x=x_batch, eps=eps,
                                        eps_iter=eps_iter,
                                        nb_iter=50,
                                        norm=np.inf,
                                        clip_min=0.0,
                                        clip_max=1.0,
                                        y=y_batch,
                                        targeted=False,
                                        decay_factor=1.0,
                                        sanity_checks=False)
        print(adv.numpy().shape)
        output = tf.concat([output,adv],0)
    print(output.numpy().shape)

    return output

def generate_adv_examples_FGSM(model,x,y,batch_size,eps):

    index = 0

    iters = x.shape[0]//batch_size
    print(iters)

    x_batch = x[index:index + batch_size]
    y_batch = y[index:index + batch_size]
    index = index + batch_size
    print(x_batch.shape)

    adv = fast_gradient_method(model_fn=model,
                                     x=x_batch, eps=eps,
                                     norm=np.inf,
                                     clip_min=0.0,
                                     clip_max=1.0,
                                     y=y_batch,
                                     targeted=False,
                                     sanity_checks=False)
    output = adv

    for i in range(1,iters,1):
        x_batch = x[index:index+batch_size]
        y_batch = y[index:index+batch_size]
        index = index + batch_size
        adv = fast_gradient_method(model_fn=model,
                                   x=x_batch, eps=eps,
                                   norm=np.inf,
                                   clip_min=0.0,
                                   clip_max=1.0,
                                   y=y_batch,
                                   targeted=False,
                                   sanity_checks=False)
        print(adv.numpy().shape)
        output = tf.concat([output,adv],0)
    print(output.numpy().shape)

    return output



def generating(model,x_train,y_train,x_test,y_test,batch_size,eps,eps_iter,dataset,model_name):


    x_adv = generate_adv_examples_FGSM(model=model, x=x_train, y=y_train, batch_size=batch_size, eps=eps)

    print(np.max(x_train - x_adv))

    np.save('data/' + dataset + '/' + model_name + '/x_train_adv_FGSM.npy', x_adv)

    model.evaluate(x_adv, y_train, verbose=2)

    x_adv = generate_adv_examples_FGSM(model=model, x=x_test, y=y_test, batch_size=batch_size, eps=eps)

    print(np.max(x_test - x_adv))

    np.save('data/' + dataset + '/' + model_name + '/x_test_adv_FGSM.npy', x_adv)

    model.evaluate(x_adv, y_test, verbose=2)

    x_adv = generate_adv_examples_MIM(model=model, x=x_train, y=y_train, batch_size=batch_size, eps=eps,
                                      eps_iter=eps_iter)

    print(np.max(x_train - x_adv))

    np.save('data/' + dataset + '/' + model_name + '/x_train_adv_MIM.npy', x_adv)

    model.evaluate(x_adv, y_train, verbose=2)

    x_adv = generate_adv_examples_MIM(model=model, x=x_test, y=y_test, batch_size=batch_size, eps=eps,
                                      eps_iter=eps_iter)

    print(np.max(x_test - x_adv))

    np.save('data/' + dataset + '/' + model_name + '/x_test_adv_MIM.npy', x_adv)

    model.evaluate(x_adv, y_test, verbose=2)

    x_adv = generate_adv_examples_PGD(model=model, x=x_train, y=y_train, batch_size=batch_size, eps=eps,
                                      eps_iter=eps_iter)

    print(np.max(x_train - x_adv))

    np.save('data/' + dataset + '/' + model_name + '/x_train_adv_PGD.npy', x_adv)

    model.evaluate(x_adv, y_train, verbose=2)

    x_adv = generate_adv_examples_PGD(model=model, x=x_test, y=y_test, batch_size=batch_size, eps=eps,
                                      eps_iter=eps_iter)

    print(np.max(x_test - x_adv))

    np.save('data/' + dataset + '/' + model_name + '/x_test_adv_PGD.npy', x_adv)

    model.evaluate(x_adv, y_test, verbose=2)

    x_adv = generate_adv_examples_BIM(model=model, x=x_train, y=y_train, batch_size=batch_size, eps=eps,
                                      eps_iter=eps_iter)

    print(np.max(x_train - x_adv))

    np.save('data/' + dataset + '/' + model_name + '/x_train_adv_BIM.npy', x_adv)

    model.evaluate(x_adv, y_train, verbose=2)

    x_adv = generate_adv_examples_BIM(model=model, x=x_test, y=y_test, batch_size=batch_size, eps=eps,
                                      eps_iter=eps_iter)

    print(np.max(x_test - x_adv))

    np.save('data/' + dataset + '/' + model_name + '/x_test_adv_BIM.npy', x_adv)

    model.evaluate(x_adv, y_test, verbose=2)


if __name__=='__main__':
    x_test = np.load('data/adversarial_examples/imagenet/x_test.npy')
    y_test = np.load('data/adversarial_examples/imagenet/y_test.npy')

    x_test = tf.convert_to_tensor(x_test,dtype=tf.float32)
    scope = [2/255., 4/255., 6/255., 8/255., 10/255.]
    file_name = [2,4,6,8,10]

    classifier,_ = models.get_imagenet_vgg16()
    for i in range(len(scope)):
        x_adv = generate_adv_examples_MIM(model=classifier,x = x_test,y=y_test,batch_size=50,eps=scope[i],eps_iter=0.001)
        np.save('data/adversarial_examples/imagenet/vgg16/different_noise_intensity/'+str(file_name[i])+'/x_test_adv_MIM',x_adv)

    classifier, _ = models.get_imagenet_mobilenet()
    for i in range(len(scope)):
        x_adv = generate_adv_examples_MIM(model=classifier,x = x_test,y=y_test,batch_size=50,eps=scope[i],eps_iter=0.001)
        np.save('data/adversarial_examples/imagenet/mobilenet/different_noise_intensity/' + str(
            file_name[i]) + '/x_test_adv_MIM', x_adv)

    classifier, _ = models.get_imagenet_inceptionv3()
    for i in range(len(scope)):
        x_adv = generate_adv_examples_MIM(model=classifier,x = x_test,y=y_test,batch_size=20,eps=scope[i],eps_iter=0.001)
        np.save('data/adversarial_examples/imagenet/inceptionv3/different_noise_intensity/' + str(
            file_name[i]) + '/x_test_adv_MIM', x_adv)

    classifier, _ = models.get_imagenet_densenet121()
    for i in range(len(scope)):
        x_adv = generate_adv_examples_MIM(model=classifier, x=x_test, y=y_test, batch_size=20, eps=scope[i],
                                          eps_iter=0.001)
        np.save('data/adversarial_examples/imagenet/densenet121/different_noise_intensity/' + str(
            file_name[i]) + '/x_test_adv_MIM', x_adv)

