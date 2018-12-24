


#def visualise_tsne(encoder, test_generator)

def visualise_latents(encoder, test_generator):
    c = []
    for col in ['r', 'c', 'y', 'k',  'b', 'g', 'm']:
        for _c in range(7):
            for i in range(nT):
                c.append(col)

    z_train = encoder.predict_generator(test_generator, steps=7*7)
    z_max = z_train
    for _ in z_train.shape:
        z_max=z_max.max(axis=-1)

    encodings = np.asarray(z_train)
    #encodings = encodings.reshape(nT * 7 * 8,latent_dim+n_y)
    plt.figure(figsize=(7, 7))
    plt.scatter(encodings[:, 0], encodings[:, 1], c=c, cmap=plt.cm.jet)
    plt.savefig('latent_viz_beta{}_latents{}_cap{}.png'.format(beta,latent_dim, capacity))
    plt.clf()

def visualise_rotations(encoder, decoder, test_generator):
# Save disentangled images (rotation interpolation)

    for si, s in enumerate(shapes):
        for i in range(7):
            im_x = next(test_generator)[0]
        la_x = im_x[2][0]
        imsave('{}.jpg'.format(s), im_x[0][0])
        z_m = encoder.predict(im_x)[0][:latent_dim]
        z_cond = np.zeros((10, latent_dim+n_y))
        for ri in range(10):
          rot = [(360/(ri+1))/360] * 3
          shape = np.zeros(8)
          shape[si+1] = 1
          z_cond[ri,:] = np.concatenate((z_m, rot, shape))

        pred = decoder.predict(z_cond)

        if not os.path.exists(os.path.join(RESULTS_DIR,'rotated_img')):
            os.mkdir(os.path.join(RESULTS_DIR,'rotated_img'))
        file_name = os.path.join(RESULTS_DIR, 'rotated_img', '{}_rotation.png'.format(s))
        save_10_images(pred, file_name)

def visualise_zinterpol(decoder, test_generator):

    im_x = next(test_generator)[0]
    z_m = encoder.predict(im_x)[0][:latent_dim]

    for target_z_index in range(latent_dim):
        z_mean2 = np.zeros((10, latent_dim))
        z_cond = np.zeros((10, latent_dim+n_y))
        for ri in range(10):
          # Change z mean value from -3.0 to +3.0
          value = -3.0 + (6.0 / 9.0) * ri

          for i in range(latent_dim):
            if( i == target_z_index ):
              z_mean2[ri][i] = value
            else:
              z_mean2[ri][i] = z_m[i]


          rot = [0.,90./360.,0.]
          shape = np.zeros(8)
          shape[np.random.randint(8)] = 1
          z_cond[ri,:] = np.concatenate((z_mean2[ri], rot, shape))

        generated_xs = decoder.predict(z_cond)

        if not os.path.exists(os.path.join(RESULTS_DIR,'disentangle_img')):
            os.mkdir(os.path.join(RESULTS_DIR,'disentangle_img'))
        file_name = os.path.join(RESULTS_DIR, 'disentangle_img', 'check_z{}.png'.format(target_z_index))
        save_10_images(generated_xs, file_name)

def save_10_images(images, file_name):
  plt.figure()
  fig, axes = plt.subplots(1, 10, figsize=(10, 1),
                           subplot_kw={'xticks': [], 'yticks': []})
  fig.subplots_adjust(hspace=0.1, wspace=0.1)

  for ax,image in zip(axes.flat, images):

    rgb_image = image.reshape((128,128,3))

    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.imshow(rgb_image)

  plt.savefig(file_name, bbox_inches='tight')
  plt.close(fig)
  plt.close()
