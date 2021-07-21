# -*- coding:utf-8 -*-
from F2M_model_V9 import *
from random import shuffle, random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import easydict
import os

FLAGS = easydict.EasyDict({"img_size": 256, 
                           
                           "load_size": 276,

                           "tar_size": 256,

                           "tar_load_size": 276,
                           
                           "batch_size": 2,
                           
                           "epochs": 200,
                           
                           "lr": 0.0002,
                           
                           "A_txt_path": "D:/[1]DB/[3]detection_DB/CelebAMask-HQ/HQ_gender/celeba_hq/train_test/female_train.txt",
                           
                           "A_img_path": "D:/[1]DB/[3]detection_DB/CelebAMask-HQ/HQ_gender/celeba_hq/train_test/female/",
                           
                           "B_txt_path": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/race_age_gender_generation/Morph_AFAD_16_63/first_fold/AFAD-F_Morph-M_40_63_16_39/train/male_16_39_train.txt",
                           
                           "B_img_path": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/Morph/All/male_16_39/",

                           "age_range": [40, 64],

                           "n_classes": 256,

                           "train": True,
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": "",
                           
                           "save_checkpoint": "",
                           
                           "sample_images": "C:/Users/Yuhwan/Pictures/img"})

g_optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)
d_optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)

def input_func(A_data, B_data):

    A_img = tf.io.read_file(A_data)
    A_img = tf.image.decode_jpeg(A_img, 3)
    A_img = tf.image.resize(A_img, [FLAGS.load_size, FLAGS.load_size])
    A_img = tf.image.random_crop(A_img, [FLAGS.img_size, FLAGS.img_size, 3])
    A_img = A_img / 127.5 - 1.

    B_img = tf.io.read_file(B_data[0])
    B_img = tf.image.decode_jpeg(B_img, 3)
    B_img = tf.image.resize(B_img, [FLAGS.tar_load_size, FLAGS.tar_load_size])
    B_img = tf.image.random_crop(B_img, [FLAGS.tar_size, FLAGS.tar_size, 3])
    B_img = B_img / 127.5 - 1.

    B_lab = int(B_data[1])

    return A_img, B_img, B_lab

#@tf.function
def model_out(model, images, training=True):
    return model(images, training=training)

def decreas_func(x):
    return tf.maximum(0, tf.math.exp(x * (-2.77 / 100)))

def increase_func(x):
    x = tf.cast(tf.maximum(1, x), tf.float32)
    return tf.math.log(x + 1e-7)

def cal_loss(A2B_G_model, B2A_G_model, B_discriminator,
             A_batch_images, B_batch_images, B_batch_labels, extract_feature_model):

    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        fake_B, real_B_mid = model_out(A2B_G_model, [A_batch_images, B_batch_images], True)
        fake_A_, fake_B_mid = model_out(B2A_G_model, [fake_B, A_batch_images], True)

        # identification    # 이것도 추가하면 괜찮지 않을까?
        #id_fake_A = model_out(B2A_G_model, [A_batch_images, B_batch_images], True)

        DB_real = model_out(B_discriminator, B_batch_images, True)
        DB_fake = model_out(B_discriminator, fake_B, True)

        ################################################################################################
        # 나이에 대한 distance를 구하는곳
        #vector_fake_B = model_out(extract_feature_model, fake_B, False)
        #vector_real_B = model_out(extract_feature_model, B_batch_images, False)
        return_loss = 0.
        for i in range(FLAGS.batch_size):
            energy_ft = tf.reduce_sum(tf.abs(fake_B_mid[i] - real_B_mid), 1)
            
            T = 4
            compare_label = tf.subtract(B_batch_labels, B_batch_labels[i])
            label_buff = tf.less(tf.abs(compare_label), T)
            label_cast = tf.cast(label_buff, tf.float32)

            realA_fakeB_loss = label_cast * increase_func(energy_ft) \
                + (1 - label_cast) * 5 * decreas_func(energy_ft)

            # A와 B 나이가 다르면 감소함수, 같으면 증가함수

            loss_buf = 0.
            for j in range(FLAGS.batch_size):
                loss_buf += realA_fakeB_loss[j]
            loss_buf /= FLAGS.batch_size

            return_loss += loss_buf
        return_loss /= FLAGS.batch_size
        ################################################################################################

        #id_loss = tf.reduce_mean(tf.abs(id_fake_A - A_batch_images)) * 10.0

        Cycle_loss = (tf.reduce_mean(tf.abs(fake_A_ - A_batch_images))) * 10.0
        G_gan_loss = tf.reduce_mean((DB_fake - tf.ones_like(DB_fake))**2)

        Adver_loss = (tf.reduce_mean((DB_real - tf.ones_like(DB_real))**2) + tf.reduce_mean((DB_fake - tf.zeros_like(DB_fake))**2)) / 2.

        g_loss = Cycle_loss + G_gan_loss + return_loss
        d_loss = Adver_loss

    g_grads = g_tape.gradient(g_loss, A2B_G_model.trainable_variables + B2A_G_model.trainable_variables)
    d_grads = d_tape.gradient(d_loss, B_discriminator.trainable_variables)

    g_optim.apply_gradients(zip(g_grads, A2B_G_model.trainable_variables + B2A_G_model.trainable_variables))
    d_optim.apply_gradients(zip(d_grads, B_discriminator.trainable_variables))

    return g_loss, d_loss

def main():
    extract_feature_model = tf.keras.applications.MobileNetV2(include_top=False,
                                                              input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    extract_feature_model.trainable = False

    h = extract_feature_model.output
    h = tf.keras.layers.GlobalAveragePooling2D()(h)
    h = tf.keras.layers.Dense(FLAGS.n_classes)(h)
    extract_feature_model = tf.keras.Model(inputs=extract_feature_model.input, outputs=h)
    extract_feature_model.summary()

    A2B_G_model = F2M_generator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    B2A_G_model = F2M_generator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    B_discriminator = F2M_discriminator(input_shape=(FLAGS.tar_size, FLAGS.tar_size, 3))

    A2B_G_model.summary()
    B_discriminator.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(A2B_G_model=A2B_G_model, B2A_G_model=B2A_G_model,
                                   B_discriminator=B_discriminator,
                                   g_optim=g_optim, d_optim=d_optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)

    if FLAGS.train:
        count = 0

        A_images = np.loadtxt(FLAGS.A_txt_path, dtype="<U100", skiprows=0, usecols=0)
        A_images = [FLAGS.A_img_path + data for data in A_images]
        A_labels = np.loadtxt(FLAGS.A_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        B_images = np.loadtxt(FLAGS.B_txt_path, dtype="<U100", skiprows=0, usecols=0)
        B_images = [FLAGS.B_img_path + data for data in B_images]
        B_labels = np.loadtxt(FLAGS.B_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        for epoch in range(FLAGS.epochs):
            min_ = min(len(A_images), len(B_images))
            shuffle(A_images)
            B = list(zip(B_images, B_labels))
            shuffle(B)
            B_images, B_labels = zip(*B)
            A_images = A_images[:min_]
            B_images = B_images[:min_]
            B_labels = B_labels[:min_]

            A_train_img = np.array(A_images)
            B_zip = np.array(list(zip(B_images, B_labels)))

            # 가까운 나이에 대해서 distance를 구하는 loss를 구성하면, 결국에는 해당이미지의 나이를 그대로 생성하는 효과?를 볼수있을것
            gener = tf.data.Dataset.from_tensor_slices((A_train_img, B_zip))
            gener = gener.shuffle(len(B_images))
            gener = gener.map(input_func)
            gener = gener.batch(FLAGS.batch_size)
            gener = gener.prefetch(tf.data.experimental.AUTOTUNE)

            train_idx = len(A_train_img) // FLAGS.batch_size
            train_it = iter(gener)
            
            for step in range(train_idx):
                A_batch_images, B_batch_images, B_batch_labels = next(train_it)

                g_loss, d_loss = cal_loss(A2B_G_model, B2A_G_model, B_discriminator,
                                          A_batch_images, B_batch_images, B_batch_labels, extract_feature_model)

                print("Epoch = {}[{}/{}];\nStep(iteration) = {}\nG_Loss = {}, D_loss = {}".format(epoch,step,train_idx,
                                                                                                  count+1,
                                                                                                  g_loss, d_loss))
                
                if count % 100 == 0:
                    fake_B, _ = model_out(A2B_G_model, [A_batch_images, B_batch_images], False)

                    plt.imsave(FLAGS.sample_images + "/fake_B_{}.jpg".format(count), fake_B[0] * 0.5 + 0.5)
                    plt.imsave(FLAGS.sample_images + "/real_B_{}.jpg".format(count), B_batch_images[0] * 0.5 + 0.5)
                    plt.imsave(FLAGS.sample_images + "/real_A_{}.jpg".format(count), A_batch_images[0] * 0.5 + 0.5)


                #if count % 1000 == 0:
                #    num_ = int(count // 1000)
                #    model_dir = "%s/%s" % (FLAGS.save_checkpoint, num_)
                #    if not os.path.isdir(model_dir):
                #        print("Make {} folder to store the weight!".format(num_))
                #        os.makedirs(model_dir)
                #    ckpt = tf.train.Checkpoint(A2B_G_model=A2B_G_model, B2A_G_model=B2A_G_model,
                #                               A_discriminator=A_discriminator, B_discriminator=B_discriminator,
                #                               g_optim=g_optim, d_optim=d_optim)
                #    ckpt_dir = model_dir + "/F2M_V8_{}.ckpt".format(count)
                #    ckpt.save(ckpt_dir)

                count += 1


if __name__ == "__main__":
    main()
