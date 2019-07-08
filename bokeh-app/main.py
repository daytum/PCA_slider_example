#!/usr/bin/env python
# coding: utf-8

# In[1]:


from intake import cat
import numpy as np
import numpy.linalg

import urllib
import os

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.models.widgets import Slider

import bokeh
from bokeh.core.validation.warnings import EMPTY_LAYOUT
bokeh.core.validation.silence(EMPTY_LAYOUT)

import panel as pn
pn.extension()

df = cat.unconv_MV.read()[['LogPerm','Por']]

scaler = MinMaxScaler(feature_range=(-1,1))

scaled_df = pd.DataFrame(scaler.fit_transform(df))
scaled_df.columns = ['Scaled LogPerm', 'Scaled Por']

eigenvalues, eigenvectors = np.linalg.eig(scaled_df.cov().values)

def remote_jupyter_proxy_url(port):
    """
    Callable to configure Bokeh's show method when a proxy must be
    configured.

    If port is None we're asking about the URL
    for the origin header.
    """
    base_url = 'https://classroom.daytum.org/'
    host = urllib.parse.urlparse(base_url).netloc

    # If port is None we're asking for the URL origin
    # so return the public hostname.
    if port is None:
        return host

    service_url_path = os.environ['JUPYTERHUB_SERVICE_PREFIX']
    proxy_url_path = 'proxy/%d' % port

    user_url = urllib.parse.urljoin(base_url, service_url_path)
    full_url = urllib.parse.urljoin(user_url, proxy_url_path)
    return full_url


def rotate(alpha):
    
    largest_eigenvector = np.dot([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]], 
                             eigenvectors[1,:])

    projected_points = ((np.dot(scaled_df.values, largest_eigenvector) / 
                     np.dot(largest_eigenvector, largest_eigenvector))[:, None] * largest_eigenvector)
    
    variance =  np.linalg.norm(projected_points - scaled_df.values, axis=1)
    
    hist, edges = np.histogram(variance, density=True, bins=50)
    
    variance_length_dict = dict(top=hist, bottom=np.zeros_like(hist), left=edges[:-1], right=edges[1:])
    
    variance_source_dict = dict(x=np.array([scaled_df.values[:,0], projected_points[:,0]]).T.tolist(),
                                y=np.array([scaled_df.values[:,1], projected_points[:,1]]).T.tolist())

    principle_component_source_dict = dict(x=np.array([-largest_eigenvector[0], largest_eigenvector[0]]) * 1.5, 
                                           y=np.array([-largest_eigenvector[1], largest_eigenvector[1]]) * 1.5)
    
    return (variance_source_dict, principle_component_source_dict, variance_length_dict)


def create_plot():
    
    lplt = figure(match_aspect=True, x_range=(-1.5, 1.5), y_range=(-1.5,1.5), 
                  toolbar_location=None, plot_height=400, plot_width=400, 
                  x_axis_label='Scaled LogPerm', y_axis_label='Scaled Porsity')
   
    hplt = figure(toolbar_location=None, plot_height=400, x_range=(0,7), y_range=(0, 1.5), 
                  x_axis_label='Number of Occurances', y_axis_label='Scaled Reconstruction Error')

    alpha = Slider(title="Rotation:", value=0.0, start=-np.pi/2., end=np.pi/2., step=0.1)

    variance_source_dict, principle_component_source_dict, variance_length_dict = rotate(alpha.value)

    variance_source = ColumnDataSource(variance_source_dict)
    principle_component_source = ColumnDataSource(principle_component_source_dict)
    variance_length_source = ColumnDataSource(variance_length_dict)
    
    lplt.circle(x='Scaled LogPerm', y='Scaled Por', source=ColumnDataSource(scaled_df))

    lplt.multi_line('x', 'y', source=variance_source, color='red')

    lplt.line('x', 'y', source=principle_component_source, color='black', line_width=3)
                                
    hplt.quad('bottom', 'top', 'left', 'right', source=variance_length_source, 
              fill_color="#036564", line_color="#033649")

    def callback(attr, old, new):
    
          variance_source.data, principle_component_source.data, variance_length_source.data = rotate(alpha.value)

    
    alpha.on_change('value', callback)
    
    
    header = pn.panel("""<a href="https://www.daytum.org"><img width=200 
        src="https://github.com/daytum/logos/blob/master/daytum_logo_2019.png?raw=true"></a>
        """, width=200)
    
    footer = pn.pane.Markdown("""The black line represents the first *principal component* of the data.
    Principal Component Analysis (PCA) is simultaneously a way of finding summerizing characteristics
    of data and giving you information to assist in predicting or reconstructing the original
    features.  The first goal (summerizing relationships between features) is performed by maximizing
    the variance along the the principal component direction.  This can be seen visually as the "spread"
    of the red lines intersecting the black line.  You can see by moving the slider bar that the original
    configuration is the one that maximizes this spread.  The black line is a characteristic of the data
    that describes the strongest differences among the original features.  <br><br>
    You might also notice that any one of the blue dots can be *reconstructed* by taking the distance
    along the black line from the origin to the intersection of the perpenticular projection from the
    blue dot to the black line (i.e. the red lines).  The red lines are indicators of *reconstruction error*
    and they are simultaneously minimized in totallity at the original configuration of the plot.  Again,
    this can be seen visually by moving the slider bar and visualing the aggregate length of the red
    lines, also you can quantitatively observe the histogram of the red line lengths on the right side
    figure.  If only the first principal component was used to reconstruct the original data, the sum of
    the squares of red lines could be used as an estimate of the reconstruction error.  Additionally, if
    this aggregate error is small enough, it may assist in *feature reduction* by allowing you to only use
    the first principal component as a predictor for the original data.
    """)
    
    return pn.Column(pn.Row(pn.Spacer(width=400),header), alpha, pn.Row(lplt, hplt), footer)


slider = pn.Pane(create_plot)
slider.servable()
#slider.app(notebook_url=remote_jupyter_proxy_url);

