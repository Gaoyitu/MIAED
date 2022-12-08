from cleverhans.tf2.attacks import carlini_wagner_l2
import numpy as np


def generate_adversarial_examples_cw2(model,x,y,batch_size,save_path):

    kwargs = {'batch_size': batch_size}
    x_adv = carlini_wagner_l2.carlini_wagner_l2(model, x, **kwargs)

    print(np.max(x - x_adv))

    model.evaluate(x_adv, y, verbose=2)

    np.save(save_path,x_adv)














