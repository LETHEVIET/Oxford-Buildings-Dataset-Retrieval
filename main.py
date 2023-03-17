import pickle
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image, ImageOps
from itertools import accumulate
import faiss
import tqdm
# import matplotlib.image as mpimg
# import matplotlib.pyplot as plt

from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform

from scipy.spatial import cKDTree

DATASETPATH = './oxbuild_images-v1/'

with open('imagesName.pkl', 'rb') as file:
    image_ls = pickle.load(file)

image_ls.sort()
image_ls = [DATASETPATH + img for img in image_ls]

delf = hub.load('https://tfhub.dev/google/delf/1').signatures['default']

def run_delf(image):
  np_image = np.array(image)
  float_image = tf.image.convert_image_dtype(np_image, tf.float32)

  return delf(
      image=float_image,
      score_threshold=tf.constant(100.0),
      image_scales=tf.constant([0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0]),
      max_feature_num=tf.constant(1000))

def load_image(path):
  image = Image.open(path) 
  image = ImageOps.fit(image, (256, 256), Image.ANTIALIAS)
  return image

with open('delf_features_new.pkl', 'rb') as file:
    features = pickle.load(file)

locations_agg = np.concatenate([features[id]['locations'] for id in range(len(features))])
descriptors_agg = np.concatenate([features[id]['descriptors'] for id in range(len(features))])
accumulated_indexes_boundaries = list(accumulate([features[id]['locations'].shape[0] for id in range(len(features))]))

oxford5k_index = faiss.IndexFlatL2(40)
oxford5k_index.add(descriptors_agg)

def retrieval(image, my_bar):
    query_image = ImageOps.fit(image, (256, 256), Image.ANTIALIAS)
    query_feature = run_delf(query_image)    
    dist, indices = oxford5k_index.search(query_feature['descriptors'].numpy(), k=2)
    
    unique_indices = np.array(list(set(indices.flatten())))
    unique_indices.sort()
    if unique_indices[-1] == descriptors_agg.shape[0]:
        unique_indices = unique_indices[:-1]

    unique_image_indexes = np.array(
        list(set([np.argmax([np.array(accumulated_indexes_boundaries)>index]) 
                  for index in unique_indices])))
    
    # dist_id = np.argsort(dist.flatten())
    # dist_idx = [ indices.flatten()[d] for d in dist_id]

    # unique_image_indexes = np.array(
    #     list(set([np.argmax([np.array(accumulated_indexes_boundaries)>index]) 
    #               for index in dist_idx])))
    # unique_image_indexes = unique_image_indexes[:20]

    error_ls = []

    distance_threshold = 0.8

    feature_1 = query_feature

    locations_1 = feature_1['locations']
    descriptors_1 = feature_1['descriptors']
    num_features_1 = locations_1.shape[0]
    d1_tree = cKDTree(descriptors_1)

    inliers_counts = []
    count = 0
    for i in tqdm.tqdm(unique_image_indexes):

      # print(i, end=' ')
      feature_2 = features[i]
      locations_2 = feature_2['locations']
      descriptors_2 = feature_2['descriptors']
      num_features_2 = locations_2.shape[0]

      _, indices = d1_tree.query(
          descriptors_2, distance_upper_bound=distance_threshold)
      
      locations_2_to_use = np.array([
          locations_2[i]
          for i in range(num_features_2)
          if indices[i] != num_features_1
      ])
      locations_1_to_use = np.array([
          locations_1[indices[i],]
          for i in range(num_features_2)
          if indices[i] != num_features_1
      ])

      try:
        _, inliers = ransac((locations_1_to_use, locations_2_to_use),
                                    AffineTransform,
                                    min_samples=3,
                                    residual_threshold=20,
                                    max_trials=1000)
      except:
        error_ls.append(i)
        inliers = []
      
      if inliers is None or len(inliers) == 0:
        continue

      inliers_counts.append({"index": i, "inliers": sum(inliers)})
      count += 1
      my_bar.progress(count / len(unique_image_indexes), text='Calculating best matches ....')
    # print(len(inliers_counts))
    # print(error_ls)
    my_bar.progress(1.0)
    top_match = sorted(inliers_counts, key=lambda k: k['inliers'], reverse=True)[:10]

    result = []
    for match in top_match:
      result.append({'name': image_ls[match['index']], 'image': Image.open(image_ls[match['index']])})

    return result

#########################################################################################

import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
st.set_option('deprecation.showfileUploaderEncoding', False)

# Upload an image and set some options for demo purposes
st.header("OXFORD5K SEARCH")
img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'])
aspect_ratio = None

if img_file:
    img = Image.open(img_file)
    
    cropped_img = st_cropper(img, realtime_update=True, box_color='#2596be',
                                aspect_ratio=aspect_ratio)
    
    if st.button('Search'):
        st.write('Selected region')
        st.image(cropped_img)
        # cropped_img
        st.write("Please wait, it may takes about 3 to 4 minutes ...")
        my_bar = st.progress(0, text='Calculating best matches ....')
        result = retrieval(cropped_img, my_bar)
        st.write("Results")
        for i in range(len(result)):
           st.write(result[i]['name'])
           st.image(result[i]['image'], width=400)

    