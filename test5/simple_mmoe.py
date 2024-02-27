import tensorflow as tf
import simple_layer


class ModelConfig:
    def __init__(self):
        self.feature_name_list = [
            'user_id', 'item_id', 'sdk_type', 'remote_host', 'device_type',
            'dtu', 'click_goods_num', 'buy_click_num', 'goods_show_num',
            'goods_click_num', 'brand_name'
        ]
        self.train_data = "../data/train/170.tfrecord"
        self.test_data = "../data/test/180.tfrecord"
        self.save_model_path = "../model/save_model"
        self.batch_size = 64
        # self.label_name = 'ctr'
        self.ctr = 'ctr'
        self.cvr = 'cvr'
        self.expert_num = 5
        self.shuffle_size = 100
        self.feature_size = 20000
        self.embedding_size = 8
        self.lr = 0.01
        self.epochs = 6


model_config = ModelConfig()
train_loss = tf.keras.metrics.Mean(name="loss")
train_ctr_auc = tf.keras.metrics.AUC(name="train_ctr_auc")
train_cvr_auc = tf.keras.metrics.AUC(name="train_cvr_auc")
test_ctr_auc = tf.keras.metrics.AUC(name="test_ctr_auc")
test_cvr_auc = tf.keras.metrics.AUC(name="test_cvr_auc")


def parse_example(example):
    feats = {model_config.ctr: tf.io.FixedLenFeature(shape=[1], dtype=tf.float32),
             model_config.cvr: tf.io.FixedLenFeature(shape=[1], dtype=tf.float32)}
    for feature_name in model_config.feature_name_list:
        feats[feature_name] = tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)
    feats = tf.io.parse_single_example(example, feats)
    return feats


def train_step(train_data, model, ctr_loss, cvr_loss, opt):
    try:
        ds = iter(train_data)
        train_loss.reset_states()
        train_ctr_auc.reset_states()
        train_cvr_auc.reset_states()
        while True:
            with tf.GradientTape() as tape:
                temp = next(ds)
                ctr_target = temp.pop(model_config.ctr)
                cvr_target = temp.pop(model_config.cvr)
                logit = model(temp)

                ctr_scala_loss = tf.reduce_sum(ctr_loss(ctr_target, logit[0]))
                cvr_scala_loss = tf.reduce_sum(cvr_loss(cvr_target, logit[1]))

                scala_loss = 0.5 * ctr_scala_loss + 0.5 * cvr_scala_loss
                g = tape.gradient(scala_loss, model.trainable_variables)
                opt.apply_gradients(zip(g, model.trainable_variables))

                train_ctr_auc.update_state(cvr_target, logit[0])
                train_cvr_auc.update_state(ctr_target, logit[1])
                train_loss.update_state(scala_loss)
    except:
        print("---")


def test_step(test_data, model):
    try:
        ds = iter(test_data)
        test_ctr_auc.reset_states()
        test_cvr_auc.reset_states()
        while True:
            temp = next(ds)
            ctr_target = temp.pop(model_config.ctr)
            cvr_target = temp.pop(model_config.cvr)
            logit = model(temp)
            test_ctr_auc.update_state(cvr_target, logit[0])
            test_cvr_auc.update_state(ctr_target, logit[1])
    except:
        print("---")


if __name__ == "__main__":
    train_data = (tf.data.TFRecordDataset(model_config.train_data)
                  .map(parse_example)
                  .shuffle(model_config.shuffle_size)
                  .batch(model_config.batch_size))

    test_data = (tf.data.TFRecordDataset(model_config.test_data)
                 .map(parse_example)
                 .batch(model_config.batch_size))

    one = next(iter(train_data))
    # print(one)
    model = simple_layer.buildMMOEModel(vars(model_config))

    opt = tf.keras.optimizers.Adam(learning_rate=model_config.lr)
    ctr_loss = tf.keras.losses.BinaryCrossentropy(name=model_config.ctr)
    cvr_loss = tf.keras.losses.BinaryCrossentropy(name=model_config.cvr)

    for epoch in range(model_config.epochs):
        train_step(train_data, model, ctr_loss, cvr_loss, opt)
        test_step(test_data, model)
        print(
            f"epoch:{epoch},loss:{train_loss.result().numpy():.4f},train_ctr_auc:{train_ctr_auc.result().numpy():.2f},train_cvr_auc:{train_cvr_auc.result().numpy():.2f}"
            f",test_ctr_auc:{test_ctr_auc.result().numpy():.2f},test_cvr_auc:{test_cvr_auc.result().numpy():.2f}")

    tf.saved_model.save(model, model_config.save_model_path)
