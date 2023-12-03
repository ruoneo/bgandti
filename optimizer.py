import numpy as np
import tensorflow as tf


class OptimizerCycle(object):
    def __init__(self, preds, labels, pos_weight, norm, d_real, d_fake, GD_real, GD_fake, preds_z2g, labels_z2g, preds_cycle, labels_cycle, gradient, gradient_z, settings):
        preds_sub = preds
        labels_sub = labels

        self.real = d_real
        self.settings = settings

        # Discrimminator Loss
        self.dc_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.real), logits=self.real, name='dclreal'))
        # self.dc_loss_real = - tf.reduce_mean(self.real)
        self.dc_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake, name='dcfake'))
        # self.dc_loss_fake = tf.reduce_mean(d_fake)
        # GP_loss = tf.reduce_mean(tf.square(tf.sqrt(tf.reduce_mean(tf.square(gradient), axis = [0, 1])) - 1))
        # GP_loss_z = tf.reduce_mean(tf.square(tf.sqrt(tf.reduce_mean(tf.square(gradient_z), axis = [0, 1])) - 1))
        # self.dc_loss = self.dc_loss_fake + self.dc_loss_real + 10.0 * GP_loss

        self.GD_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(GD_real), logits=GD_real, name='GD_real'))
        # self.GD_loss_real = - tf.reduce_mean(GD_real)
        self.GD_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(GD_fake), logits=GD_fake, name='GD_fake'))
        # self.GD_loss_fake = tf.reduce_mean(GD_fake)

        self.dc_loss = self.dc_loss_fake + self.dc_loss_real
        self.GD_loss = self.GD_loss_fake + self.GD_loss_real

        # Generator loss
        generator_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake, name='gl'))
        # generator_loss = -self.dc_loss_fake
        generator_loss_z2g = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(GD_fake), logits=GD_fake, name='G_z2g'))
        # generator_loss_z2g = -self.GD_loss_fake
        # pos_weight，允许人们通过向上或向下加权相对于负误差的正误差的成本来权衡召回率和精确度
        self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))

        cost_cycle = norm * tf.reduce_mean(tf.square(preds_cycle - labels_cycle))

        cost_z2g = norm * tf.reduce_mean(tf.square(preds_z2g - labels_z2g))

        self.cost = self.cost + cost_cycle
        self.generator_loss = generator_loss + self.cost
        self.generator_loss_z2g = generator_loss_z2g

        all_variables = tf.trainable_variables()
        dc_var = [var for var in all_variables if 'dc_' in var.name]
        en_var = [var for var in all_variables if 'e_' in var.name]
        GG_var = [var for var in all_variables if 'GG' in var.name]
        GD_var = [var for var in all_variables if 'GD' in var.name]

        with tf.variable_scope(tf.get_variable_scope()):
            self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=self.settings.discriminator_learning_rate,
                                                                  beta1=0.9, name='adam1').minimize(self.dc_loss,
                                                                                                    var_list=dc_var)  # minimize(dc_loss_real, var_list=dc_var)

            self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=self.settings.discriminator_learning_rate,
                                                              beta1=0.9, name='adam2').minimize(self.generator_loss, var_list=en_var)

            self.discriminator_optimizer_z2g = tf.train.AdamOptimizer(learning_rate=self.settings.discriminator_learning_rate,
                                                                      beta1=0.9, name='adam1').minimize(self.GD_loss, var_list=GD_var)

            self.generator_optimizer_z2g = tf.train.AdamOptimizer(learning_rate=self.settings.discriminator_learning_rate,
                                                                  beta1=0.9, name='adam2').minimize(self.generator_loss_z2g, var_list=GG_var)

        # 值得注意的是，这个地方，除了对抗优化之外，
        # 还单纯用cost损失又优化了一遍，
        # 待会儿看训练的时候注意看是在哪部分进行的这部分优化操作
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.settings.learning_rate)  # Adam Optimizer
        self.opt_op = self.optimizer.minimize(self.cost)
        # self.grads_vars = self.optimizer.compute_gradients(self.cost)

        # self.optimizer_z2g = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        # self.opt_op_z2g = self.optimizer.minimize(cost_z2g)
        # self.grads_vars_z2g = self.optimizer.compute_gradients(cost_z2g)


class Optimizer(object):
    def __init__(self, model, model_z2g, D_Graph, discriminator, placeholders, pos_weight, norm, d_real, num_nodes, GD_real, settings):
        self.opt = self.construct_optimizer(model, model_z2g, D_Graph, discriminator, placeholders, pos_weight, norm, d_real, num_nodes, GD_real, settings)

    def construct_optimizer(self, model, model_z2g, D_Graph, discriminator, placeholders, pos_weight, norm, d_real, num_nodes, GD_real, settings):
        z2g = model_z2g.construct()
        hidden = z2g[1]
        z2g = z2g[0]
        preds_z2g = model.construct(hidden=hidden, reuse=True)[0]
        g2z = model.construct()

        embeddings = g2z[0]
        reconstructions = g2z[1]
        d_fake = discriminator.construct(embeddings, reuse=True)
        GD_fake = D_Graph.construct(z2g, reuse=True)

        epsilon = tf.random_uniform(shape=[1], minval=0.0, maxval=1.0)
        interpolated_input = epsilon * placeholders['real_distribution'] + (1 - epsilon) * embeddings
        gradient = tf.gradients(discriminator.construct(interpolated_input, reuse=True), [interpolated_input])[0]

        epsilon = tf.random_uniform(shape=[1], minval=0.0, maxval=1.0)
        interpolated_input = epsilon * placeholders['features_dense'] + (1 - epsilon) * z2g
        gradient_z = tf.gradients(D_Graph.construct(interpolated_input, reuse=True), [interpolated_input])[0]

        opt = OptimizerCycle(preds=reconstructions,
                             labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False), [-1]),
                             pos_weight=pos_weight,
                             norm=norm,
                             d_real=d_real,
                             d_fake=d_fake,
                             GD_real=GD_real,
                             GD_fake=GD_fake,
                             preds_z2g=preds_z2g,
                             labels_z2g=placeholders['real_distribution'],
                             preds_cycle=model_z2g.construct(embeddings, reuse=True)[0],
                             labels_cycle=placeholders['features_dense'],
                             gradient=gradient,
                             gradient_z=gradient_z,
                             settings=settings)
        return opt

def construct_feed_dict(adj_normalized, adj, features, placeholders):
    # construct feed dictionary
    # .update()用法就是将()内的字段增加到dict当中
    feed_dict = dict()  # 创建一个空字典
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict

def update(model, opt, sess, adj_norm, adj_label, features, placeholders, adj, distribution, adj_dense, settings):
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: settings.dropout})
    feed_dict.update({placeholders['features_dense']: adj_dense})
    feed_dict.update({placeholders['dropout']: 0})
    z_real_dist = np.random.randn(adj.shape[0], settings.hidden2)
    z_real_dist = distribution.sample(adj.shape[0])
    feed_dict.update({placeholders['real_distribution']: z_real_dist})

    for j in range(5):
        _, reconstruct_loss = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
    g_loss, _ = sess.run([opt.generator_loss, opt.generator_optimizer], feed_dict=feed_dict)
    d_loss, _ = sess.run([opt.dc_loss, opt.discriminator_optimizer], feed_dict=feed_dict)

    GD_loss, _ = sess.run([opt.GD_loss, opt.discriminator_optimizer_z2g], feed_dict=feed_dict)
    GG_loss, _ = sess.run([opt.generator_loss_z2g, opt.generator_optimizer_z2g], feed_dict=feed_dict)
    # GD_loss = sess.run(opt.GD_loss, feed_dict=feed_dict)
    # GG_loss = sess.run(opt.generator_loss_z2g, feed_dict=feed_dict)
    # g_loss = sess.run(opt.generator_loss, feed_dict=feed_dict)
    # d_loss = sess.run(opt.dc_loss, feed_dict=feed_dict)
    emb = sess.run(model.z_mean, feed_dict=feed_dict)
    avg_cost = [reconstruct_loss, d_loss, g_loss, GD_loss, GG_loss]

    return emb, avg_cost