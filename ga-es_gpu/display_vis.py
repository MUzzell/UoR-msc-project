__copyright__ = """
Copyright (c) 2018 Uber Technologies, Inc.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import argparse
import time
import math
import os
import tensorflow as tf
import numpy as np
import gym
import pickle
import tabular_logger as tlogger
import gym_tensorflow
from ga import TrainingState, Offspring, OffspringCached
from neuroevolution.helper import SharedNoiseTable, make_schedule
from PIL import Image
from neuroevolution.models import LargeModel, Model, ModelVirtualBN
from tf_cnnvis import activation_visualization
import cv2 as cv

default_seeds = [157822315, [94987453, 0.002], [61990409, 0.002], [132377995, 0.002], [126312029, 0.002], [93915238, 0.002], [204022435, 0.002], [171168059, 0.002], [95856784, 0.002], [205934773, 0.002], [213365167, 0.002], [56944619, 0.002], [130129199, 0.002], [97653261, 0.002], [218695493, 0.002], [28585353, 0.002], [88260057, 0.002], [64456571, 0.002], [98751337, 0.002], [87617692, 0.002], [125110843, 0.002], [152209542, 0.002], [23777454, 0.002], [118715026, 0.002], [99788230, 0.002], [75625082, 0.002], [159513938, 0.002], [49484131, 0.002], [212507985, 0.002], [67766136, 0.002], [105190117, 0.002], [33338001, 0.002], [91160896, 0.002], [95386222, 0.002], [45411355, 0.002], [35330570, 0.002], [52225337, 0.002], [165107533, 0.002], [168561753, 0.002], [227083606, 0.002], [214214551, 0.002], [149424426, 0.002], [227684991, 0.002], [35940913, 0.002], [37453011, 0.002], [47170722, 0.002], [92046206, 0.002], [133306577, 0.002], [241955088, 0.002], [41258860, 0.002], [124242631, 0.002], [238064391, 0.002], [46235460, 0.002], [202890570, 0.002], [162416334, 0.002], [78853643, 0.002], [46547745, 0.002], [42268049, 0.002], [202162794, 0.002], [7635563, 0.002], [157757570, 0.002], [237930574, 0.002], [136918954, 0.002], [74723244, 0.002], [2358695, 0.002], [186515303, 0.002], [123109724, 0.002], [109957783, 0.002], [139233438, 0.002], [149436411, 0.002], [8346966, 0.002], [50835889, 0.002], [88695187, 0.002], [211719991, 0.002], [7283371, 0.002], [187750894, 0.002], [154620515, 0.002], [1567632, 0.002], [152631412, 0.002], [38971002, 0.002], [210627707, 0.002], [13311476, 0.002], [157351125, 0.002], [141462178, 0.002], [77606659, 0.002], [22653392, 0.002], [126720849, 0.002], [103503555, 0.002], [138904418, 0.002], [35877598, 0.002], [144448095, 0.002], [143072590, 0.002], [22256859, 0.002], [136674067, 0.002], [54962461, 0.002], [204771663, 0.002], [126594400, 0.002], [143362648, 0.002], [160053218, 0.002], [36505, 0.002], [234586339, 0.002], [8689386, 0.002], [65244214, 0.002], [39252740, 0.002], [64390487, 0.002], [191138142, 0.002], [114738239, 0.002], [184992944, 0.002], [178848289, 0.002], [685758, 0.002], [3946484, 0.002], [9120869, 0.002], [77891561, 0.002], [21685013, 0.002], [38580333, 0.002], [116730475, 0.002], [235053809, 0.002], [227204700, 0.002], [3795447, 0.002], [81764102, 0.002], [166797679, 0.002], [243641394, 0.002], [100513946, 0.002], [99241225, 0.002], [52990995, 0.002], [184304246, 0.002], [46027535, 0.002], [231862778, 0.002], [213237946, 0.002], [227474205, 0.002], [158534897, 0.002], [121346355, 0.002], [63714427, 0.002], [243338063, 0.002], [77546631, 0.002], [178281288, 0.002], [220770449, 0.002], [145968980, 0.002], [29894061, 0.002], [127519509, 0.002], [77760912, 0.002], [61219600, 0.002], [161595533, 0.002], [221480691, 0.002], [206642829, 0.002], [215721178, 0.002], [229794882, 0.002], [31325752, 0.002], [224755578, 0.002], [21220559, 0.002], [171553173, 0.002], [145243964, 0.002], [210190857, 0.002], [150615695, 0.002], [86169422, 0.002], [68813648, 0.002], [107799990, 0.002], [55892198, 0.002], [2389691, 0.002], [181991246, 0.002], [226957512, 0.002], [17909594, 0.002], [54447626, 0.002], [43646598, 0.002], [235297721, 0.002], [193625953, 0.002], [102087733, 0.002], [90041055, 0.002], [76368893, 0.002], [142359520, 0.002], [46114189, 0.002], [80413082, 0.002], [215509948, 0.002], [224115155, 0.002], [85931155, 0.002], [178125002, 0.002], [212925031, 0.002], [18694268, 0.002], [46238885, 0.002], [84948476, 0.002], [8914603, 0.002], [167599874, 0.002], [187802420, 0.002], [170522346, 0.002], [219794607, 0.002], [138665107, 0.002], [157723712, 0.002], [198373356, 0.002], [17916877, 0.002], [149620586, 0.002], [171324275, 0.002], [33574148, 0.002], [438145, 0.002], [30578731, 0.002], [111771703, 0.002], [215725985, 0.002], [226048734, 0.002], [159650006, 0.002], [94154665, 0.002], [33938839, 0.002], [147816297, 0.002], [55752950, 0.002], [217323253, 0.002], [5963619, 0.002], [236473711, 0.002], [133530026, 0.002], [31605617, 0.002], [176598781, 0.002], [117344669, 0.002], [236439401, 0.002], [232750544, 0.002], [126125063, 0.002], [20500196, 0.002], [156839548, 0.002], [17602010, 0.002], [92471651, 0.002], [92360499, 0.002], [7769454, 0.002], [136213779, 0.002], [118114719, 0.002], [105398561, 0.002], [131436589, 0.002], [202193758, 0.002], [60385109, 0.002], [179870277, 0.002], [239557330, 0.002], [187854329, 0.002], [45710618, 0.002], [186771058, 0.002], [189689540, 0.002], [212594973, 0.002], [13689343, 0.002], [20117487, 0.002], [141338221, 0.002], [174004389, 0.002], [49948893, 0.002], [121246710, 0.002], [80925692, 0.002], [39571786, 0.002], [181570823, 0.002], [181260602, 0.002], [179666712, 0.002], [157724327, 0.002], [142152925, 0.002], [72763175, 0.002], [124426367, 0.002], [95423105, 0.002], [142795024, 0.002], [149481164, 0.002], [156867918, 0.002], [193305436, 0.002], [225062969, 0.002], [51384529, 0.002], [153485310, 0.002], [186021802, 0.002], [126854908, 0.002], [57495392, 0.002], [93191535, 0.002], [123655689, 0.002], [204221002, 0.002], [147627388, 0.002], [100922671, 0.002], [43042488, 0.002], [109793369, 0.002], [86175815, 0.002], [103521806, 0.002]]

RUNS = 3
VIDEO_SIZE = 1024


def combine_single_layer(im, viz, layer, pad):

    assert layer in viz
    image = viz[layer][0].astype(np.uint8)

    w = im.shape[1] + pad + image.shape[1]
    h = np.max([im.shape[0], image.shape[0]])

    im_cat = np.zeros((h, w, im.shape[2]), dtype=np.uint8)

    im_cat[:im.shape[0], :im.shape[1]] = im
    im_cat[:image.shape[0], im.shape[1]+pad:w, 0] = image
    im_cat[:image.shape[0], im.shape[1]+pad:w, 1] = image
    im_cat[:image.shape[0], im.shape[1]+pad:w, 2] = image

    return im_cat

def combine_viz(im, viz, layer=None, pad=2):

    if layer != 'all':
        return combine_single_layer(im, viz, layer, pad)

    # import pdb; pdb.set_trace()
    n = int(math.ceil(math.sqrt(len(viz))))

    h_max = np.max([viz[k][0].shape[0] for k in viz.keys()])
    w_max = np.max([viz[k][0].shape[1] for k in viz.keys()])

    w = im.shape[1] + h_max * n + (n * pad)
    h = np.max([im.shape[0], w_max * n + (n * pad)])

    im_cat = np.zeros((h, w, im.shape[2]),
                      dtype=np.uint8)

    im_cat[:im.shape[0], :im.shape[1]] = im
    x = im.shape[1] + pad
    y = 0

    c = 0
    keys = sorted([*viz])
    for a in range(n):
        for b in range(n):
            if c >= len(keys):
                continue
            image = viz[keys[c]][0].astype(np.uint8)
            # FIX: there has to be a better way!
            im_cat[y:y+image.shape[0], x:x+image.shape[1], 0] = image
            im_cat[y:y+image.shape[0], x:x+image.shape[1], 1] = image
            im_cat[y:y+image.shape[0], x:x+image.shape[1], 2] = image

            x += image.shape[1] + pad
            c += 1

        y += h_max + pad
        x = im.shape[1] + pad

    return im_cat


def handle_frame(im, outvid, viewer, game, mut_count, add_text=False):
    assert outvid.isOpened()
    im = cv.resize(im, dsize=(VIDEO_SIZE, VIDEO_SIZE), interpolation=cv.INTER_LINEAR)

    if add_text:
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(im, 'Game: {}'.format(game),
                   (8, 225), font, 0.3, (255, 255, 255))
        cv.putText(im, 'Mutations: {}'.format(mut_count),
                   (8, 240), font, 0.3, (255, 255, 255))
    if outvid:
        outvid.write(im)
    if viewer:
        viewer.imshow(im)


def get_model(model_name):
    if model_name == "Model":
        return Model()
    if model_name == "LargeModel":
        return LargeModel()
    if model_name == "ModelVirtualBN":
        return ModelVirtualBN()

    raise NotImplemented()


def get_nn_images(sess, input_op, model):

    run_input = {input_op: sess.run(input_op)}
    if hasattr(model, 'ref_batch_idx'):
        run_input[model.ref_batch_idx] = 0

    return activation_visualization(
        sess, run_input,
        input_tensor=input_op, layers='r',
        path_logdir='/dev/null',
        path_outdir='/dev/null')


def main(game, filename=None, out_dir=None, model_name='LargeModel',
         add_text=False, num_runs=RUNS, layer=None):

    seeds = default_seeds
    outvid = None
    viewer = None
    iteration = None
    state = None

    if filename:
        with open(filename, 'rb+') as file:
            state = pickle.load(file)
            #if hasattr(state, 'best_score'):
            #    seeds = state.best_score.seeds
            #    iteration = len(seeds)
            #    print("Loading GA snapshot from best_score, iteration: ", len(seeds))
            if hasattr(state, 'elite'):
                seeds = state.elite.seeds
                iteration = state.it
                print("Loading GA snapshot from elite, iteration: {} / {}".format(len(seeds), iteration))
            else:
                seeds = None
                iteration = state.it
                print("Loading ES snapshot, iteration: {}", state.it)

    fourcc = cv.VideoWriter_fourcc(*'H264')

    env = gym_tensorflow.make(game, 1)

    model = get_model(model_name)
    obs_op = env.observation()
    reset_op = env.reset()

    if model.requires_ref_batch:
        def make_env(b):
            return gym_tensorflow.make(game=game, batch_size=1)
        with tf.Session() as sess:
            ref_batch = gym_tensorflow.get_ref_batch(make_env, sess, 128)
            ref_batch = ref_batch[:, ...]
    else:
        ref_batch = None

    input_op = tf.expand_dims(obs_op, axis=1)
    action_op = model.make_net(input_op, env.action_space, batch_size=1, ref_batch=ref_batch)
    if env.discrete_action:
        action_op = tf.argmax(action_op, axis=-1, output_type=tf.int32)
    rew_op, done_op = env.step(action_op)

    out_vids = {'all': cv.VideoWriter(os.path.join(out_dir, 'all.mp4'),
                                      fourcc, 16, (VIDEO_SIZE, VIDEO_SIZE))}

    if hasattr(env.unwrapped, 'render'):
        obs_op = env.unwrapped.render()

        def display_obs(im, viz):
            # pdb.set_trace()
            if im.shape[1] > 1:
                im = np.bitwise_or(im[0, 0, ...], im[0, 1, ...])
            else:
                im = im[0, 0, ...]
            for key in out_vids.keys():
                im = combine_viz(im, viz, key)
                handle_frame(im, out_vids[key], viewer, game, iteration, add_text)
    else:
        def display_obs(im, viz):
            im = im[0, :, :, -1]
            im = np.stack([im] * 3, axis=-1)
            im = (im * 255).astype(np.uint8)
            for key in out_vids.keys():
                im = combine_viz(im, viz, key)
                handle_frame(im, out_vids[key], viewer, game, iteration, add_text)

    rewards = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.initialize()
        tlogger.info(model.description)

        if seeds:
            noise = SharedNoiseTable()
            weights = model.compute_weights_from_seeds(noise, seeds)
            model.load(sess, 0, weights, seeds)
        else:
            weights = state.theta
            model.load(sess, 0, weights, (weights, 0))

        success, images = get_nn_images(sess, input_op, model)

        for key in images.keys():
            out_vids[key] = cv.VideoWriter(
                os.path.join(out_dir, '{}.mp4'.format(key.replace('/', '-'))),
                fourcc, 16, (VIDEO_SIZE, VIDEO_SIZE))

        for i in range(num_runs):
            sess.run(reset_op)
            # recorder.capture_frame()

            total_rew = 0
            num_frames = 0
            while True:
                img = sess.run(obs_op)
                success, images = get_nn_images(sess, input_op, model)

                rew, done = sess.run([rew_op, done_op])
                num_frames += 1
                total_rew += rew[0]
                display_obs(img, images)
                # time.sleep(4/60)
                if done[0] or num_frames == 50:
                    rewards += [total_rew]
                    print('Final reward: ', total_rew, 'after', num_frames, 'steps')
                    break

    print(rewards)
    print("Mean: ", np.mean(rewards))
    print("Std: ", np.std(rewards))

    for key in out_vids:
        out_vids[key].release()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("game", default="frostbite")
    ap.add_argument("file", default=None)
    ap.add_argument('-m', '--model', default='LargeModel')
    ap.add_argument('-o', "--out_dir", default='/tmp/')
    ap.add_argument("-t", "--add_text", action="store_true", default=False)
    ap.add_argument("-n", "--num_runs", default=RUNS, type=int)
    ap.add_argument("-s", "--min_score", default=0, type=int)
    ap.add_argument("-l", "--layer", default=None)
    args = ap.parse_args()
    main(args.game, args.file, args.out_dir, args.model, args.add_text,
         args.num_runs, args.layer)
