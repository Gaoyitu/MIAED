
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from functools import partial
from basics import *


class Autoencoder(tf.keras.Model):
    def __init__(self,encoder,fc_model,decoder):
        super(Autoencoder, self).__init__()

        self.encoder = encoder
        self.fc_model = fc_model
        self.decoder = decoder

        self.encoder.summary()
        self.fc_model.summary()
        self.decoder.summary()


    def train_step(self,x):

        with tf.GradientTape(persistent=True) as t:
            x_encoder = self.encoder(x,training=True)
            x_vector = self.fc_model(x_encoder,training=True)
            outputs = self.decoder(x_vector,training=True)

            loss = self.compiled_loss(x,outputs)
        grad_encoder = t.gradient(loss, self.encoder.trainable_variables)
        grad_fc_model = t.gradient(loss, self.fc_model.trainable_variables)
        grad_decoder = t.gradient(loss, self.decoder.trainable_variables)


        self.optimizer.apply_gradients(zip(grad_encoder, self.encoder.trainable_variables))
        self.optimizer.apply_gradients(zip(grad_fc_model,self.fc_model.trainable_variables))
        self.optimizer.apply_gradients(zip(grad_decoder, self.decoder.trainable_variables))

        return {m.name: m.result() for m in self.metrics}


class MI_Autoencoder():
    def __init__(self,conf,encoder,fc_model,decoder,discriminator_global,discriminator_local,discriminator_prior):

        self.inputH = conf.imageH
        self.inputW = conf.imageW
        self.channel = conf.channel
        self.epochs = conf.epochs
        self.batch_size = conf.batch_size
        self.opt = AdamOptWrapper(learning_rate=conf.learning_rate, beta_1=conf.beta1)
        self.gpw = conf.gpw
        self.rec_w = conf.rec_w
        self.global_w = conf.global_w
        self.local_w = conf.local_w
        self.dw = conf.dw

        self.encoder = encoder
        self.fc_model = fc_model
        self.decoder = decoder
        self.discriminator_global = discriminator_global
        self.discriminator_local = discriminator_local
        self.discriminator_prior = discriminator_prior


        self.encoder.summary()
        self.fc_model.summary()
        self.decoder.summary()
        self.discriminator_global.summary()
        self.discriminator_local.summary()
        self.discriminator_prior.summary()




    def train_MI_autoencoder(self,x,v):
        x = tf.data.Dataset.from_tensor_slices(x)
        x = x.batch(self.batch_size)


        a_train_loss = tf.keras.metrics.Mean()
        d_train_loss = tf.keras.metrics.Mean()

        for epoch in range(self.epochs):
            for x_batch in zip(x):
                print('epoch:', epoch)

                loss_d,gp = self.train_discriminator(x_batch,v)
                d_train_loss(loss_d)

                loss_total = self.train_autoencoder(x_batch)
                a_train_loss(loss_total)

                a_train_loss.reset_states()
                d_train_loss.reset_states()


    def train_autoencoder(self, x):

        _,batch_s,_,_,_ = np.array(x).shape

        x_shuffle = tf.random.shuffle(tf.reshape(x,(batch_s,self.inputH,self.inputW,self.channel)))

        with tf.GradientTape(persistent=True) as t:
            x_feature = self.encoder(x, training=True)
            x_shuffle_feature = self.encoder(x_shuffle, training=True)

            _,h,w,c = x_feature.shape
            x_vector = self.fc_model(x_feature,training=True)
            _,l = x_vector.shape


            x_flatten = tf.reshape(x_feature,(-1,h*w*c))

            x_shuffle_flatten = tf.reshape(x_shuffle_feature,(-1,h*w*c))

            pos_global = tf.concat([x_flatten,x_vector],axis=1)
            neg_global = tf.concat([x_shuffle_flatten,x_vector],axis=1)


            pos_global_logits = self.discriminator_global(pos_global,training=True)
            neg_global_logits = self.discriminator_global(neg_global,training=True)

            loss_global = global_loss(pos=pos_global_logits,neg=neg_global_logits)

            x_vector_local = tf.tile(x_vector,[1,h*w])
            x_vector_local = tf.reshape(x_vector_local,(-1,h,w,l))

            pos_local = tf.concat([x_feature,x_vector_local],axis=3)
            neg_local = tf.concat([x_shuffle_feature,x_vector_local],axis=3)

            pos_local_logits = self.discriminator_local(pos_local,training=True)
            neg_local_logits = self.discriminator_local(neg_local,training=True)

            loss_local = local_loss(pos=pos_local_logits, neg=neg_local_logits)


            x_vector_noised = tf.add(x_vector, tf.random.normal(shape=x_vector.shape, mean=0.0, stddev=0.1))

            x_rec = self.decoder(x_vector, training=True)
            x_rec_noised = self.decoder(x_vector_noised, training=True)

            loss_rec = MSE(x, x_rec)
            loss_rec_noised = MSE(x_rec, x_rec_noised)

            x_fake_logits = self.discriminator_prior(x_vector,training=True)

            loss_d = generator_loss(x_fake_logits)

            print('loss_rec:',loss_rec.numpy())
            print('loss_rec_noised:',loss_rec_noised.numpy())
            print('loss_global:',loss_global.numpy())
            print('loss_local:',loss_local.numpy())
            print('loss_d:',loss_d.numpy())

            loss_total = self.rec_w*loss_rec +self.rec_w*loss_rec_noised +self.global_w*loss_global+self.local_w*loss_local + self.dw*loss_d

        grad_encoder = t.gradient(loss_total, self.encoder.trainable_variables)
        grad_fc_model = t.gradient(loss_total, self.fc_model.trainable_variables)
        grad_decoder = t.gradient(loss_total, self.decoder.trainable_variables)
        grad_discriminator_global = t.gradient(loss_total,self.discriminator_global.trainable_variables)
        grad_discriminator_local = t.gradient(loss_total,self.discriminator_local.trainable_variables)


        self.opt.apply_gradients(zip(grad_encoder, self.encoder.trainable_variables))
        self.opt.apply_gradients(zip(grad_fc_model, self.fc_model.trainable_variables))
        self.opt.apply_gradients(zip(grad_decoder, self.decoder.trainable_variables))
        self.opt.apply_gradients(zip(grad_discriminator_global,self.discriminator_global.trainable_variables))
        self.opt.apply_gradients(zip(grad_discriminator_local,self.discriminator_local.trainable_variables))

        return loss_total


    def train_discriminator(self, x,vector_dimension):


        _, batch_s, _, _, _ = np.array(x).shape

        real = np.random.randn(batch_s, vector_dimension) * 5.

        with tf.GradientTape() as t:
            fake = self.fc_model(self.encoder(x,training=True),training=True)

            fake_logits = self.discriminator_prior(fake, training=True)

            real_logits = self.discriminator_prior(real, training=True)
            cost = discriminator_loss(fake_logits, real_logits)
            gp = gradient_penalty(batch_size=batch_s,f=partial(self.discriminator_prior, training=True), real=real, fake=fake)
            cost += self.gpw * gp

            print('d: cost:',cost.numpy())
            print('d:gp:',gp.numpy())
        grad = t.gradient(cost, self.discriminator_prior.trainable_variables)
        self.opt.apply_gradients(zip(grad, self.discriminator_prior.trainable_variables))
        return cost, gp

