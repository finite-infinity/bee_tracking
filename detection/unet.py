import tensorflow as tf
if int(tf.__version__[0]) > 1:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
import math

#图像大小不变 卷积
def _create_conv_relu(inputs, name, filters, dropout_ratio, is_training, strides=[1,1], kernel_size=[3,3], padding="SAME", relu=True):
    net = tf.layers.conv2d(inputs=inputs, filters=filters, strides=strides, kernel_size=kernel_size, padding=padding, name="%s_conv" % name)
    if dropout_ratio > 0:
        net = tf.layers.dropout(inputs=net, rate=dropout_ratio, training=is_training, name="%s_dropout" % name)
    net = tf.layers.batch_normalization(net, center=True, scale=False, training=is_training, name="%s_bn" % name)
    if relu:
        net = tf.nn.relu(net) # leaky relu
    return net


def _create_pool(data, name, pool_size=[2,2], strides=[2,2]):
    pool = tf.layers.max_pooling2d(inputs=data, pool_size=pool_size, strides=strides, padding='SAME', name=name)
    return pool

#缩小 UNet前半段，图像变小（池化），通道变多
def _contracting_path(data, num_layers, num_filters, dropout_ratio, is_training):
    interim = []

    dim_out = num_filters  #输出维度
    for i in range(num_layers):   #批量层
        name = "c_%i" % i
        conv1 = _create_conv_relu(data, name + "_1", dim_out, dropout_ratio=dropout_ratio, is_training=is_training)
        conv2 = _create_conv_relu(conv1, name + "_2", dim_out, dropout_ratio=dropout_ratio, is_training=is_training)
        pool = _create_pool(conv2, name) #下采样
        data = pool

        dim_out *=2  
        interim.append(conv2)

    return (interim, data)

#扩张 UNet后半段 ，图像（interim）变大（反卷积）
def _expansive_path(data, interim, num_layers, dim_in, dropout_ratio, is_training):
    dim_out = int(dim_in / 2)
    for i in range(num_layers):
        name = "e_%i" % i
        #反卷积，filters是反卷积后得到的特征图数量（补0后卷积，大小与卷积前相当）
        upconv = tf.layers.conv2d_transpose(data, filters=dim_out, kernel_size=2, strides=2, name="%s_upconv" % name)
        concat = tf.concat([interim[len(interim)-i-1], upconv], 3)   #在通道维度上concat上缩小层的倒数第i-1层 （大小不一样就剪裁）
        conv1 = _create_conv_relu(concat, name + "_1", dim_out, dropout_ratio=dropout_ratio, is_training=is_training)
        #suffix = "last" if (i == num_layers - 1) else suffix + "_2"
        conv2 = _create_conv_relu(conv1, name + "_2", dim_out, dropout_ratio=dropout_ratio, is_training=is_training)
        data = conv2
        dim_out = int(dim_out / 2)
    return data


def create_unet2(num_layers, num_filters, data, is_training, prev=None, dropout_ratio=0, classes=3):

    (interim, contracting_data) = _contracting_path(data, num_layers, num_filters, dropout_ratio, is_training) #缩小层结果
    #中间层
    middle_dim = num_filters * 2**num_layers   #缩小完的data维度
    middle_conv_1 = _create_conv_relu(contracting_data, "m_1", middle_dim, dropout_ratio=dropout_ratio, is_training=is_training)
    middle_conv_2 = _create_conv_relu(middle_conv_1, "m_2", middle_dim, dropout_ratio=dropout_ratio, is_training=is_training)
    middle_end = middle_conv_2 

    expansive_path = _expansive_path(middle_end, interim, num_layers, middle_dim, dropout_ratio, is_training) #扩张层结果
    last_relu = expansive_path
    
    #这是偏置？
    if prev != None:
        expansive_path = tf.concat([prev, expansive_path], 3)

    conv_logits = _create_conv_relu(expansive_path, "conv_logits", num_filters, dropout_ratio=dropout_ratio, is_training=is_training)
    logits = _create_conv_relu(conv_logits, "logits", classes, dropout_ratio=dropout_ratio, is_training=is_training)  #输出三个特征

    conv_angle = _create_conv_relu(expansive_path, "conv_angle", num_filters, dropout_ratio=dropout_ratio, is_training=is_training, relu=False)
    angle_pred = _create_conv_relu(conv_angle, "angle_pred", 1, dropout_ratio=dropout_ratio, is_training=is_training, relu=False) #输出一个角度（不经relu）
    return logits, last_relu, angle_pred  #last_relu是没经过最后两层卷积预测的UNet最后一层


#计算加权且经概率化的类loss
def loss(logits, labels, weight_map, numclasses=3):
    oh_labels = tf.one_hot(indices=tf.cast(labels, tf.uint8), depth=numclasses, name="one_hot") #one_hot label
    loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=oh_labels)  #logits转化为概率（softmax），再计算交叉熵损失（正好label独热了）
    weighted_loss = tf.multiply(loss_map, weight_map)  #给loss加权
    loss = tf.reduce_mean(weighted_loss, name="weighted_loss")
    #tf.add_to_collection('losses', loss)
    return loss #tf.add_n(tf.get_collection('losses'), name='total_loss')


def angle_loss(angle_pred, angle_labels, weight_map):

    sh = tf.shape(angle_pred)
    angle_pred = tf.reshape(angle_pred, [sh[0],sh[1],sh[2]])
    #掩膜
    bg_mask = tf.logical_or(tf.less(angle_pred, 0), tf.less(angle_labels, 0)) #只要两个中有一个角度<0，就返回True(向后掩膜）
    fg_mask = tf.logical_not(bg_mask)    #bg_mask的反向（向前掩膜）
    
    #通过mask过滤weight_map和angle，将前（后）向的loss单独求出来（不然loss值符号相反）
    #loss = （ sin( (angle_pred - angle_label） *pi) )^2
    fg_loss = tf.multiply(tf.boolean_mask(weight_map, fg_mask), 
                          tf.square(tf.sin((tf.boolean_mask(angle_pred, fg_mask) - tf.boolean_mask(angle_labels, fg_mask))*math.pi)))
    bg_loss = tf.multiply(tf.boolean_mask(weight_map, bg_mask),
                          tf.square(tf.boolean_mask(angle_pred, bg_mask) - tf.boolean_mask(angle_labels, bg_mask)))

    fg_loss = tf.reduce_mean(fg_loss, name="weighted_angle_loss")
    bg_loss = tf.reduce_mean(bg_loss, name="weighted_bg_angle_loss")

    loss = fg_loss + bg_loss
    #tf.add_to_collection('losses', loss)
    return loss #tf.add_n(tf.get_collection('losses'), name='total_loss')
