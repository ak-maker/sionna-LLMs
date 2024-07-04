
# Ray Tracing<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#ray-tracing" title="Permalink to this headline"></a>
    
This module provides a differentiable ray tracer for radio propagation modeling.
The best way to get started is by having a look at the <a class="reference external" href="../examples/Sionna_Ray_Tracing_Introduction.html">Sionna Ray Tracing Tutorial</a>.
The <a class="reference external" href="../em_primer.html">Primer on Electromagnetics</a> provides useful background knowledge and various definitions that are used throughout the API documentation.
    
The most important component of the ray tracer is the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene" title="sionna.rt.Scene">`Scene`</a>.
It has methods for the computation of propagation <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a> (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.compute_paths" title="sionna.rt.Scene.compute_paths">`compute_paths()`</a>) and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap" title="sionna.rt.CoverageMap">`CoverageMap`</a> (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.coverage_map" title="sionna.rt.Scene.coverage_map">`coverage_map()`</a>).
Sionna has several integrated <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#example-scenes">Example Scenes</a> that you can use for your own experiments. In this <a class="reference external" href="https://youtu.be/7xHLDxUaQ7c">video</a>, we explain how you can create your own scenes using <a class="reference external" href="https://www.openstreetmap.org">OpenStreetMap</a> and <a class="reference external" href="https://www.blender.org">Blender</a>.
You can preview a scene within a Jupyter notebook (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.preview" title="sionna.rt.Scene.preview">`preview()`</a>) or render it to a file from the viewpoint of a camera (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.render" title="sionna.rt.Scene.render">`render()`</a> or <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.render_to_file" title="sionna.rt.Scene.render_to_file">`render_to_file()`</a>).
    
Propagation <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a> can be transformed into time-varying channel impulse responses (CIRs) via <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.cir" title="sionna.rt.Paths.cir">`cir()`</a>. The CIRs can then be used for link-level simulations in Sionna via the functions <a class="reference internal" href="channel.wireless.html#sionna.channel.cir_to_time_channel" title="sionna.channel.cir_to_time_channel">`cir_to_time_channel()`</a> or <a class="reference internal" href="channel.wireless.html#sionna.channel.cir_to_ofdm_channel" title="sionna.channel.cir_to_ofdm_channel">`cir_to_ofdm_channel()`</a>. Alternatively, you can create a dataset of CIRs that can be used by a channel model with the help of <a class="reference internal" href="channel.wireless.html#sionna.channel.CIRDataset" title="sionna.channel.CIRDataset">`CIRDataset`</a>.
    
The paper <a class="reference external" href="https://nvlabs.github.io/sionna/made_with_sionna.html#sionna-rt-differentiable-ray-tracing-for-radio-propagation-modeling">Sionna RT: Differentiable Ray Tracing for Radio Propagation Modeling</a> shows how differentiable ray tracing can be used for various optimization tasks. The related <a class="reference external" href="https://nvlabs.github.io/sionna/made_with_sionna.html#sionna-rt-differentiable-ray-tracing-for-radio-propagation-modeling">notebooks</a> can be a good starting point for your own experiments.

## Scene<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#scene" title="Permalink to this headline"></a>
    
The scene contains everything that is needed for radio propagation simulation
and rendering.
    
A scene is a collection of multiple instances of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.SceneObject" title="sionna.rt.SceneObject">`SceneObject`</a> which define
the geometry and materials of the objects in the scene.
The scene also includes transmitters (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a>) and receivers (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a>)
for which propagation <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a> or  channel impulse responses (CIRs) can be computed,
as well as cameras (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a>) for rendering.
    
A scene is loaded from a file using the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.load_scene" title="sionna.rt.load_scene">`load_scene()`</a> function.
Sionna contains a few <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#example-scenes">Example Scenes</a>.
The following code snippet shows how to load one of them and
render it through the lens of the preconfigured scene <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a> “scene-cam-0”:
```python
scene = load_scene(sionna.rt.scene.munich)
scene.render(camera="scene-cam-0")
```

<img alt="../_images/munich.png" src="https://nvlabs.github.io/sionna/_images/munich.png" />
    
You can preview a scene in an interactive 3D viewer within a Jupyter notebook using <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.preview" title="sionna.rt.Scene.preview">`preview()`</a>:
```python
scene.preview()
```

    
In the code snippet above, the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.load_scene" title="sionna.rt.load_scene">`load_scene()`</a> function returns the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene" title="sionna.rt.Scene">`Scene`</a> instance which can be used
to access scene objects, transmitters, receivers, cameras, and to set the
frequency for radio wave propagation simulation. Note that you can load only a single scene at a time.
    
It is important to understand that all transmitters in a scene share the same <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.AntennaArray" title="sionna.rt.AntennaArray">`AntennaArray`</a> which can be set
through the scene property <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.tx_array" title="sionna.rt.Scene.tx_array">`tx_array`</a>. The same holds for all receivers whose <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.AntennaArray" title="sionna.rt.AntennaArray">`AntennaArray`</a>
can be set through <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.rx_array" title="sionna.rt.Scene.rx_array">`rx_array`</a>. However, each transmitter and receiver can have a different position and orientation.
    
The code snippet below shows how to configure the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.tx_array" title="sionna.rt.Scene.tx_array">`tx_array`</a> and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.rx_array" title="sionna.rt.Scene.rx_array">`rx_array`</a> and
to instantiate a transmitter and receiver.
```python
# Configure antenna array for all transmitters
scene.tx_array = PlanarArray(num_rows=8,
                          num_cols=2,
                          vertical_spacing=0.7,
                          horizontal_spacing=0.5,
                          pattern="tr38901",
                          polarization="VH")
# Configure antenna array for all receivers
scene.rx_array = PlanarArray(num_rows=1,
                          num_cols=1,
                          vertical_spacing=0.5,
                          horizontal_spacing=0.5,
                          pattern="dipole",
                          polarization="cross")
# Create transmitter
tx = Transmitter(name="tx",
              position=[8.5,21,27],
              orientation=[0,0,0])
scene.add(tx)
# Create a receiver
rx = Receiver(name="rx",
           position=[45,90,1.5],
           orientation=[0,0,0])
scene.add(rx)
# TX points towards RX
tx.look_at(rx)
print(scene.transmitters)
print(scene.receivers)
```

```python
{'tx': <sionna.rt.transmitter.Transmitter object at 0x7f83d0555d30>}
{'rx': <sionna.rt.receiver.Receiver object at 0x7f81f00ef0a0>}
```

    
Once you have loaded a scene and configured transmitters and receivers, you can use the scene method
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.compute_paths" title="sionna.rt.Scene.compute_paths">`compute_paths()`</a> to compute propagation paths:
```python
paths = scene.compute_paths()
```

    
The output of this function is an instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a> and can be used to compute channel
impulse responses (CIRs) using the method <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.cir" title="sionna.rt.Paths.cir">`cir()`</a>.
You can visualize the paths within a scene by one of the following commands:
```python
scene.preview(paths=paths) # Open preview showing paths
scene.render(camera="preview", paths=paths) # Render scene with paths from preview camera
scene.render_to_file(camera="preview",
                     filename="scene.png",
                     paths=paths) # Render scene with paths to file
```

<img alt="../_images/paths_visualization.png" src="https://nvlabs.github.io/sionna/_images/paths_visualization.png" />
    
Note that the calls to the render functions in the code above use the “preview” camera which is configured through
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.preview" title="sionna.rt.Scene.preview">`preview()`</a>. You can use any other <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a> that you create here as well.
    
The function <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.coverage_map" title="sionna.rt.Scene.coverage_map">`coverage_map()`</a> computes a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap" title="sionna.rt.CoverageMap">`CoverageMap`</a> for every transmitter in a scene:
```python
cm = scene.coverage_map(cm_cell_size=[1.,1.], # Configure size of each cell
                        num_samples=1e7) # Number of rays to trace
```

    
Coverage maps can be visualized in the same way as propagation paths:
```python
scene.preview(coverage_map=cm) # Open preview showing coverage map
scene.render(camera="preview", coverage_map=cm) # Render scene with coverage map
scene.render_to_file(camera="preview",
                     filename="scene.png",
                     coverage_map=cm) # Render scene with coverage map to file
```

<img alt="../_images/coverage_map_visualization.png" src="https://nvlabs.github.io/sionna/_images/coverage_map_visualization.png" />

### Scene<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#id1" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.rt.``Scene`<a class="reference internal" href="../_modules/sionna/rt/scene.html#Scene">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene" title="Permalink to this definition"></a>
    
The scene contains everything that is needed for radio propagation simulation
and rendering.
    
A scene is a collection of multiple instances of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.SceneObject" title="sionna.rt.SceneObject">`SceneObject`</a> which define
the geometry and materials of the objects in the scene.
The scene also includes transmitters (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a>) and receivers (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a>)
for which propagation <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a>, channel impulse responses (CIRs) or coverage maps (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap" title="sionna.rt.CoverageMap">`CoverageMap`</a>) can be computed,
as well as cameras (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a>) for rendering.
    
The only way to instantiate a scene is by calling `load_scene()`.
Note that only a single scene can be loaded at a time.
    
Example scenes can be loaded as follows:
```python
scene = load_scene(sionna.rt.scene.munich)
scene.preview()
```

<img alt="../_images/scene_preview.png" src="https://nvlabs.github.io/sionna/_images/scene_preview.png" />

`add`(<em class="sig-param">`item`</em>)<a class="reference internal" href="../_modules/sionna/rt/scene.html#Scene.add">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.add" title="Permalink to this definition"></a>
    
Adds a transmitter, receiver, radio material, or camera to the scene.
    
If a different item with the same name as `item` is already part of the scene,
an error is raised.
Input
    
**item** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial" title="sionna.rt.RadioMaterial">`RadioMaterial`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a>) – Item to add to the scene




<em class="property">`property` </em>`cameras`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.cameras" title="Permalink to this definition"></a>
    
Dictionary
of cameras in the scene
Type
    
<cite>dict</cite> (read-only), { “name”, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a>}




<em class="property">`property` </em>`center`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.center" title="Permalink to this definition"></a>
    
Get the center of the scene
Type
    
[3], tf.float




<em class="property">`property` </em>`dtype`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.dtype" title="Permalink to this definition"></a>
    
Datatype used in tensors
Type
    
<cite>tf.complex64 | tf.complex128</cite>




<em class="property">`property` </em>`frequency`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.frequency" title="Permalink to this definition"></a>
    
Get/set the carrier frequency [Hz]
    
Setting the frequency updates the parameters of frequency-dependent
radio materials. Defaults to 3.5e9.
Type
    
float




`get`(<em class="sig-param">`name`</em>)<a class="reference internal" href="../_modules/sionna/rt/scene.html#Scene.get">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.get" title="Permalink to this definition"></a>
    
Returns a scene object, transmitter, receiver, camera, or radio material
Input
    
**name** (<em>str</em>) – Name of the item to retrieve

Output
    
**item** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.SceneObject" title="sionna.rt.SceneObject">`SceneObject`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial" title="sionna.rt.RadioMaterial">`RadioMaterial`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a> | <cite>None</cite>) – Retrieved item. Returns <cite>None</cite> if no corresponding item was found in the scene.




<em class="property">`property` </em>`objects`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.objects" title="Permalink to this definition"></a>
    
Dictionary
of scene objects
Type
    
<cite>dict</cite> (read-only), { “name”, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.SceneObject" title="sionna.rt.SceneObject">`SceneObject`</a>}




<em class="property">`property` </em>`radio_material_callable`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.radio_material_callable" title="Permalink to this definition"></a>
    
Get/set a callable that computes the radio material properties at the
points of intersection between the rays and the scene objects.
    
If set, then the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial" title="sionna.rt.RadioMaterial">`RadioMaterial`</a> of the objects are
not used and the callable is invoked instead to obtain the
electromagnetic properties required to simulate the propagation of radio
waves.
    
If not set, i.e., <cite>None</cite> (default), then the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial" title="sionna.rt.RadioMaterial">`RadioMaterial`</a> of objects are used to simulate the
propagation of radio waves in the scene.
    
This callable is invoked on batches of intersection points.
It takes as input the following tensors:
 
- `object_id` (<cite>[batch_dims]</cite>, <cite>int</cite>) : Integers uniquely identifying the intersected objects
- `points` (<cite>[batch_dims, 3]</cite>, <cite>float</cite>) : Positions of the intersection points

    
The callable must output a tuple/list of the following tensors:
 
- `complex_relative_permittivity` (<cite>[batch_dims]</cite>, <cite>complex</cite>) : Complex relative permittivities $\eta$ <a class="reference internal" href="../em_primer.html#equation-eta">(9)</a>
- `scattering_coefficient` (<cite>[batch_dims]</cite>, <cite>float</cite>) : Scattering coefficients $S\in[0,1]$ <a class="reference internal" href="../em_primer.html#equation-scattering-coefficient">(37)</a>
- `xpd_coefficient` (<cite>[batch_dims]</cite>, <cite>float</cite>) : Cross-polarization discrimination coefficients $K_x\in[0,1]$ <a class="reference internal" href="../em_primer.html#equation-xpd">(39)</a>. Only relevant for the scattered field.

    
**Note:** The number of batch dimensions is not necessarily equal to one.


<em class="property">`property` </em>`radio_materials`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.radio_materials" title="Permalink to this definition"></a>
    
Dictionary
of radio materials
Type
    
<cite>dict</cite> (read-only), { “name”, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial" title="sionna.rt.RadioMaterial">`RadioMaterial`</a>}




<em class="property">`property` </em>`receivers`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.receivers" title="Permalink to this definition"></a>
    
Dictionary
of receivers in the scene
Type
    
<cite>dict</cite> (read-only), { “name”, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a>}




`remove`(<em class="sig-param">`name`</em>)<a class="reference internal" href="../_modules/sionna/rt/scene.html#Scene.remove">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.remove" title="Permalink to this definition"></a>
    
Removes a transmitter, receiver, camera, or radio material from the
scene.
    
In the case of a radio material, it must not be used by any object of
the scene.
Input
    
**name** (<em>str</em>) – Name of the item to remove




<em class="property">`property` </em>`rx_array`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.rx_array" title="Permalink to this definition"></a>
    
Get/set the antenna array used by
all receivers in the scene. Defaults to <cite>None</cite>.
Type
    
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.AntennaArray" title="sionna.rt.AntennaArray">`AntennaArray`</a>




<em class="property">`property` </em>`scattering_pattern_callable`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.scattering_pattern_callable" title="Permalink to this definition"></a>
    
Get/set a callable that computes the scattering pattern at the
points of intersection between the rays and the scene objects.
    
If set, then the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial.scattering_pattern" title="sionna.rt.RadioMaterial.scattering_pattern">`scattering_pattern`</a> of
the radio materials of the objects are not used and the callable is invoked
instead to evaluate the scattering pattern required to simulate the
propagation of diffusely reflected radio waves.
    
If not set, i.e., <cite>None</cite> (default), then the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial.scattering_pattern" title="sionna.rt.RadioMaterial.scattering_pattern">`scattering_pattern`</a> of the objects’
radio materials are used to simulate the propagation of diffusely
reflected radio waves in the scene.
    
This callable is invoked on batches of intersection points.
It takes as input the following tensors:
 
- `object_id` (<cite>[batch_dims]</cite>, <cite>int</cite>) : Integers uniquely identifying the intersected objects
- `points` (<cite>[batch_dims, 3]</cite>, <cite>float</cite>) : Positions of the intersection points
- `k_i` (<cite>[batch_dims, 3]</cite>, <cite>float</cite>) : Unitary vector corresponding to the direction of incidence in the scene’s global coordinate system
- `k_s` (<cite>[batch_dims, 3]</cite>, <cite>float</cite>) : Unitary vector corresponding to the direction of the diffuse reflection in the scene’s global coordinate system
- `n` (<cite>[batch_dims, 3]</cite>, <cite>float</cite>) : Unitary vector corresponding to the normal to the surface at the intersection point

    
The callable must output the following tensor:
 
- `f_s` (<cite>[batch_dims]</cite>, <cite>float</cite>) : The scattering pattern evaluated for the previous inputs

    
**Note:** The number of batch dimensions is not necessarily equal to one.


<em class="property">`property` </em>`size`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.size" title="Permalink to this definition"></a>
    
Get the size of the scene, i.e., the size of the
axis-aligned minimum bounding box for the scene
Type
    
[3], tf.float




<em class="property">`property` </em>`synthetic_array`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.synthetic_array" title="Permalink to this definition"></a>
    
Get/set if the antenna arrays are applied synthetically.
Defaults to <cite>True</cite>.
Type
    
bool




<em class="property">`property` </em>`transmitters`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.transmitters" title="Permalink to this definition"></a>
    
Dictionary
of transmitters in the scene
Type
    
<cite>dict</cite> (read-only), { “name”, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a>}




<em class="property">`property` </em>`tx_array`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.tx_array" title="Permalink to this definition"></a>
    
Get/set the antenna array used by
all transmitters in the scene. Defaults to <cite>None</cite>.
Type
    
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.AntennaArray" title="sionna.rt.AntennaArray">`AntennaArray`</a>




<em class="property">`property` </em>`wavelength`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.wavelength" title="Permalink to this definition"></a>
    
Wavelength [m]
Type
    
float (read-only)




### compute_paths<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#compute-paths" title="Permalink to this headline"></a>

`sionna.rt.Scene.``compute_paths`(<em class="sig-param">`self`</em>, <em class="sig-param">`max_depth``=``3`</em>, <em class="sig-param">`method``=``'fibonacci'`</em>, <em class="sig-param">`num_samples``=``1000000`</em>, <em class="sig-param">`los``=``True`</em>, <em class="sig-param">`reflection``=``True`</em>, <em class="sig-param">`diffraction``=``False`</em>, <em class="sig-param">`scattering``=``False`</em>, <em class="sig-param">`scat_keep_prob``=``0.001`</em>, <em class="sig-param">`edge_diffraction``=``False`</em>, <em class="sig-param">`check_scene``=``True`</em>, <em class="sig-param">`scat_random_phases``=``True`</em>, <em class="sig-param">`testing``=``False`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.compute_paths" title="Permalink to this definition"></a>
    
Computes propagation paths
    
This function computes propagation paths between the antennas of
all transmitters and receivers in the current scene.
For each propagation path $i$, the corresponding channel coefficient
$a_i$ and delay $\tau_i$, as well as the
angles of departure $(\theta_{\text{T},i}, \varphi_{\text{T},i})$
and arrival $(\theta_{\text{R},i}, \varphi_{\text{R},i})$ are returned.
For more detail, see <a class="reference internal" href="../em_primer.html#equation-h-final">(26)</a>.
Different propagation phenomena, such as line-of-sight, reflection, diffraction,
and diffuse scattering can be individually enabled/disabled.
    
If the scene is configured to use synthetic arrays
(<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.synthetic_array" title="sionna.rt.Scene.synthetic_array">`synthetic_array`</a> is <cite>True</cite>), transmitters and receivers
are modelled as if they had a single antenna located at their
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter.position" title="sionna.rt.Transmitter.position">`position`</a>. The channel responses for each
individual antenna of the arrays are then computed “synthetically” by applying
appropriate phase shifts. This reduces the complexity significantly
for large arrays. Time evolution of the channel coefficients can be simulated with
the help of the function <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.apply_doppler" title="sionna.rt.Paths.apply_doppler">`apply_doppler()`</a> of the returned
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a> object.
    
The path computation consists of two main steps as shown in the below figure.
<img alt="../_images/compute_paths.svg" src="https://nvlabs.github.io/sionna/_images/compute_paths.svg" />
    
For a configured <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene" title="sionna.rt.Scene">`Scene`</a>, the function first traces geometric propagation paths
using <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.trace_paths" title="sionna.rt.Scene.trace_paths">`trace_paths()`</a>. This step is independent of the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial" title="sionna.rt.RadioMaterial">`RadioMaterial`</a> of the scene objects as well as the transmitters’ and receivers’
antenna <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Antenna.patterns" title="sionna.rt.Antenna.patterns">`patterns`</a> and  <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter.orientation" title="sionna.rt.Transmitter.orientation">`orientation`</a>,
but depends on the selected propagation
phenomena, such as reflection, scattering, and diffraction. The traced paths
are then converted to EM fields by the function <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.compute_fields" title="sionna.rt.Scene.compute_fields">`compute_fields()`</a>.
The resulting <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a> object can be used to compute channel
impulse responses via <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.cir" title="sionna.rt.Paths.cir">`cir()`</a>. The advantage of separating path tracing
and field computation is that one can study the impact of different radio materials
by executing <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.compute_fields" title="sionna.rt.Scene.compute_fields">`compute_fields()`</a> multiple times without
re-tracing the propagation paths. This can for example speed-up the calibration of scene parameters
by several orders of magnitude.
<p class="rubric">Example
```python
import sionna
from sionna.rt import load_scene, Camera, Transmitter, Receiver, PlanarArray
# Load example scene
scene = load_scene(sionna.rt.scene.munich)
# Configure antenna array for all transmitters
scene.tx_array = PlanarArray(num_rows=8,
                          num_cols=2,
                          vertical_spacing=0.7,
                          horizontal_spacing=0.5,
                          pattern="tr38901",
                          polarization="VH")
# Configure antenna array for all receivers
scene.rx_array = PlanarArray(num_rows=1,
                          num_cols=1,
                          vertical_spacing=0.5,
                          horizontal_spacing=0.5,
                          pattern="dipole",
                          polarization="cross")
# Create transmitter
tx = Transmitter(name="tx",
              position=[8.5,21,27],
              orientation=[0,0,0])
scene.add(tx)
# Create a receiver
rx = Receiver(name="rx",
           position=[45,90,1.5],
           orientation=[0,0,0])
scene.add(rx)
# TX points towards RX
tx.look_at(rx)
# Compute paths
paths = scene.compute_paths()
# Open preview showing paths
scene.preview(paths=paths, resolution=[1000,600])
```

<img alt="../_images/paths_preview.png" src="https://nvlabs.github.io/sionna/_images/paths_preview.png" />

Input
 
- **max_depth** (<em>int</em>) – Maximum depth (i.e., number of bounces) allowed for tracing the
paths. Defaults to 3.
- **method** (<em>str (“exhaustive”|”fibonacci”)</em>) – Ray tracing method to be used.
The “exhaustive” method tests all possible combinations of primitives.
This method is not compatible with scattering.
The “fibonacci” method uses a shoot-and-bounce approach to find
candidate chains of primitives. Initial ray directions are chosen
according to a Fibonacci lattice on the unit sphere. This method can be
applied to very large scenes. However, there is no guarantee that
all possible paths are found.
Defaults to “fibonacci”.
- **num_samples** (<em>int</em>) – Number of rays to trace in order to generate candidates with
the “fibonacci” method.
This number is split equally among the different transmitters
(when using synthetic arrays) or transmit antennas (when not using
synthetic arrays).
This parameter is ignored when using the exhaustive method.
Tracing more rays can lead to better precision
at the cost of increased memory requirements.
Defaults to 1e6.
- **los** (<em>bool</em>) – If set to <cite>True</cite>, then the LoS paths are computed.
Defaults to <cite>True</cite>.
- **reflection** (<em>bool</em>) – If set to <cite>True</cite>, then the reflected paths are computed.
Defaults to <cite>True</cite>.
- **diffraction** (<em>bool</em>) – If set to <cite>True</cite>, then the diffracted paths are computed.
Defaults to <cite>False</cite>.
- **scattering** (<em>bool</em>) – if set to <cite>True</cite>, then the scattered paths are computed.
Only works with the Fibonacci method.
Defaults to <cite>False</cite>.
- **scat_keep_prob** (<em>float</em>) – Probability with which a scattered path is kept.
This is helpful to reduce the number of computed scattered
paths, which might be prohibitively high in some scenes.
Must be in the range (0,1). Defaults to 0.001.
- **edge_diffraction** (<em>bool</em>) – If set to <cite>False</cite>, only diffraction on wedges, i.e., edges that
connect two primitives, is considered.
Defaults to <cite>False</cite>.
- **check_scene** (<em>bool</em>) – If set to <cite>True</cite>, checks that the scene is well configured before
computing the paths. This can add a significant overhead.
Defaults to <cite>True</cite>.
- **scat_random_phases** (<em>bool</em>) – If set to <cite>True</cite> and if scattering is enabled, random uniform phase
shifts are added to the scattered paths.
Defaults to <cite>True</cite>.
- **testing** (<em>bool</em>) – If set to <cite>True</cite>, then additional data is returned for testing.
Defaults to <cite>False</cite>.


Output
    
paths : <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a> – Simulated paths



### trace_paths<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#trace-paths" title="Permalink to this headline"></a>

`sionna.rt.Scene.``trace_paths`(<em class="sig-param">`self`</em>, <em class="sig-param">`max_depth``=``3`</em>, <em class="sig-param">`method``=``'fibonacci'`</em>, <em class="sig-param">`num_samples``=``1000000`</em>, <em class="sig-param">`los``=``True`</em>, <em class="sig-param">`reflection``=``True`</em>, <em class="sig-param">`diffraction``=``False`</em>, <em class="sig-param">`scattering``=``False`</em>, <em class="sig-param">`scat_keep_prob``=``0.001`</em>, <em class="sig-param">`edge_diffraction``=``False`</em>, <em class="sig-param">`check_scene``=``True`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.trace_paths" title="Permalink to this definition"></a>
    
Computes the trajectories of the paths by shooting rays
    
The EM fields corresponding to the traced paths are not computed.
They can be computed using <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.compute_fields" title="sionna.rt.Scene.compute_fields">`compute_fields()`</a>:
```python
traced_paths = scene.trace_paths()
paths = scene.compute_fields(*traced_paths)
```

    
Path tracing is independent of the radio materials, antenna patterns,
and radio device orientations.
Therefore, a set of traced paths could be reused for different values
of these quantities, e.g., to calibrate the ray tracer.
This can enable significant resource savings as path tracing is
typically significantly more resource-intensive than field computation.
    
Note that <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.compute_paths" title="sionna.rt.Scene.compute_paths">`compute_paths()`</a> does both path tracing and
field computation.
Input
 
- **max_depth** (<em>int</em>) – Maximum depth (i.e., number of interaction with objects in the scene)
allowed for tracing the paths.
Defaults to 3.
- **method** (<em>str (“exhaustive”|”fibonacci”)</em>) – Method to be used to list candidate paths.
The “exhaustive” method tests all possible combination of primitives as
paths. This method is not compatible with scattering.
The “fibonacci” method uses a shoot-and-bounce approach to find
candidate chains of primitives. Initial ray directions are arranged
in a Fibonacci lattice on the unit sphere. This method can be
applied to very large scenes. However, there is no guarantee that
all possible paths are found.
Defaults to “fibonacci”.
- **num_samples** (<em>int</em>) – Number of random rays to trace in order to generate candidates.
A large sample count may exhaust GPU memory.
Defaults to 1e6. Only needed if `method` is “fibonacci”.
- **los** (<em>bool</em>) – If set to <cite>True</cite>, then the LoS paths are computed.
Defaults to <cite>True</cite>.
- **reflection** (<em>bool</em>) – If set to <cite>True</cite>, then the reflected paths are computed.
Defaults to <cite>True</cite>.
- **diffraction** (<em>bool</em>) – If set to <cite>True</cite>, then the diffracted paths are computed.
Defaults to <cite>False</cite>.
- **scattering** (<em>bool</em>) – If set to <cite>True</cite>, then the scattered paths are computed.
Only works with the Fibonacci method.
Defaults to <cite>False</cite>.
- **scat_keep_prob** (<em>float</em>) – Probability with which to keep scattered paths.
This is helpful to reduce the number of scattered paths computed,
which might be prohibitively high in some setup.
Must be in the range (0,1).
Defaults to 0.001.
- **edge_diffraction** (<em>bool</em>) – If set to <cite>False</cite>, only diffraction on wedges, i.e., edges that
connect two primitives, is considered.
Defaults to <cite>False</cite>.
- **check_scene** (<em>bool</em>) – If set to <cite>True</cite>, checks that the scene is well configured before
computing the paths. This can add a significant overhead.
Defaults to <cite>True</cite>.


Output
 
- **spec_paths** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a>) – Computed specular paths
- **diff_paths** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a>) – Computed diffracted paths
- **scat_paths** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a>) – Computed scattered paths
- **spec_paths_tmp** (`PathsTmpData`) – Additional data required to compute the EM fields of the specular
paths
- **diff_paths_tmp** (`PathsTmpData`) – Additional data required to compute the EM fields of the diffracted
paths
- **scat_paths_tmp** (`PathsTmpData`) – Additional data required to compute the EM fields of the scattered
paths




### compute_fields<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#compute-fields" title="Permalink to this headline"></a>

`sionna.rt.Scene.``compute_fields`(<em class="sig-param">`self`</em>, <em class="sig-param">`spec_paths`</em>, <em class="sig-param">`diff_paths`</em>, <em class="sig-param">`scat_paths`</em>, <em class="sig-param">`spec_paths_tmp`</em>, <em class="sig-param">`diff_paths_tmp`</em>, <em class="sig-param">`scat_paths_tmp`</em>, <em class="sig-param">`check_scene``=``True`</em>, <em class="sig-param">`scat_random_phases``=``True`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.compute_fields" title="Permalink to this definition"></a>
    
Computes the EM fields corresponding to traced paths
    
Paths can be traced using <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.trace_paths" title="sionna.rt.Scene.trace_paths">`trace_paths()`</a>.
This method can then be used to finalize the paths calculation by
computing the corresponding fields:
```python
traced_paths = scene.trace_paths()
paths = scene.compute_fields(*traced_paths)
```

    
Paths tracing is independent from the radio materials, antenna patterns,
and radio devices orientations.
Therefore, a set of traced paths could be reused for different values
of these quantities, e.g., to calibrate the ray tracer.
This can enable significant resource savings as paths tracing is
typically significantly more resource-intensive than field computation.
    
Note that <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.compute_paths" title="sionna.rt.Scene.compute_paths">`compute_paths()`</a> does both tracing and
field computation.
Input
 
- **spec_paths** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a>) – Specular paths
- **diff_paths** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a>) – Diffracted paths
- **scat_paths** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a>) – Scattered paths
- **spec_paths_tmp** (`PathsTmpData`) – Additional data required to compute the EM fields of the specular
paths
- **diff_paths_tmp** (`PathsTmpData`) – Additional data required to compute the EM fields of the diffracted
paths
- **scat_paths_tmp** (`PathsTmpData`) – Additional data required to compute the EM fields of the scattered
paths
- **check_scene** (<em>bool</em>) – If set to <cite>True</cite>, checks that the scene is well configured before
computing the paths. This can add a significant overhead.
Defaults to <cite>True</cite>.
- **scat_random_phases** (<em>bool</em>) – If set to <cite>True</cite> and if scattering is enabled, random uniform phase
shifts are added to the scattered paths.
Defaults to <cite>True</cite>.


Output
    
**paths** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a>) – Computed paths



### coverage_map<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#coverage-map" title="Permalink to this headline"></a>

`sionna.rt.Scene.``coverage_map`(<em class="sig-param">`self`</em>, <em class="sig-param">`rx_orientation``=``(0.0,` `0.0,` `0.0)`</em>, <em class="sig-param">`max_depth``=``3`</em>, <em class="sig-param">`cm_center``=``None`</em>, <em class="sig-param">`cm_orientation``=``None`</em>, <em class="sig-param">`cm_size``=``None`</em>, <em class="sig-param">`cm_cell_size``=``(10.0,` `10.0)`</em>, <em class="sig-param">`combining_vec``=``None`</em>, <em class="sig-param">`precoding_vec``=``None`</em>, <em class="sig-param">`num_samples``=``2000000`</em>, <em class="sig-param">`los``=``True`</em>, <em class="sig-param">`reflection``=``True`</em>, <em class="sig-param">`diffraction``=``False`</em>, <em class="sig-param">`scattering``=``False`</em>, <em class="sig-param">`edge_diffraction``=``False`</em>, <em class="sig-param">`check_scene``=``True`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.coverage_map" title="Permalink to this definition"></a>
    
This function computes a coverage map for every transmitter in the scene.
    
For a given transmitter, a coverage map is a rectangular surface with
arbitrary orientation subdivded
into rectangular cells of size $\lvert C \rvert = \texttt{cm_cell_size[0]} \times  \texttt{cm_cell_size[1]}$.
The parameter `cm_cell_size` therefore controls the granularity of the map.
The coverage map associates with every cell $(i,j)$ the quantity

$$
b_{i,j} = \frac{1}{\lvert C \rvert} \int_{C_{i,j}} \lvert h(s) \rvert^2 ds
$$
    
where $\lvert h(s) \rvert^2$ is the squared amplitude
of the path coefficients $a_i$ at position $s=(x,y)$,
the integral is over the cell $C_{i,j}$, and
$ds$ is the infinitesimal small surface element
$ds=dx \cdot dy$.
The dimension indexed by $i$ ($j$) corresponds to the $y\, (x)$-axis of the
coverage map in its local coordinate system.
    
For specularly and diffusely reflected paths, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#equation-cm-def">(43)</a> can be rewritten as an integral over the directions
of departure of the rays from the transmitter, by substituting $s$
with the corresponding direction $\omega$:

$$
b_{i,j} = \frac{1}{\lvert C \rvert} \int_{\Omega} \lvert h\left(s(\omega) \right) \rvert^2 \frac{r(\omega)^2}{\lvert \cos{\alpha(\omega)} \rvert} \mathbb{1}_{\left\{ s(\omega) \in C_{i,j} \right\}} d\omega
$$
    
where the integration is over the unit sphere $\Omega$, $r(\omega)$ is the length of
the path with direction of departure $\omega$, $s(\omega)$ is the point
where the path with direction of departure $\omega$ intersects the coverage map,
$\alpha(\omega)$ is the angle between the coverage map normal and the direction of arrival
of the path with direction of departure $\omega$,
and $\mathbb{1}_{\left\{ s(\omega) \in C_{i,j} \right\}}$ is the function that takes as value
one if $s(\omega) \in C_{i,j}$ and zero otherwise.
Note that $ds = \frac{r(\omega)^2 d\omega}{\lvert \cos{\alpha(\omega)} \rvert}$.
    
The previous integral is approximated through Monte Carlo sampling by shooting $N$ rays
with directions $\omega_n$ arranged as a Fibonacci lattice on the unit sphere around the transmitter,
and bouncing the rays on the intersected objects until the maximum depth (`max_depth`) is reached or
the ray bounces out of the scene.
At every intersection with an object of the scene, a new ray is shot from the intersection which corresponds to either
specular reflection or diffuse scattering, following a Bernoulli distribution with parameter the
squared scattering coefficient.
When diffuse scattering is selected, the direction of the scattered ray is uniformly sampled on the half-sphere.
The resulting Monte Carlo estimate is:

$$
\hat{b}_{i,j}^{\text{(ref)}} = \frac{4\pi}{N\lvert C \rvert} \sum_{n=1}^N \lvert h\left(s(\omega_n)\right)  \rvert^2 \frac{r(\omega_n)^2}{\lvert \cos{\alpha(\omega_n)} \rvert} \mathbb{1}_{\left\{ s(\omega_n) \in C_{i,j} \right\}}.
$$
    
For the diffracted paths, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#equation-cm-def">(43)</a> can be rewritten for any wedge
with length $L$ and opening angle $\Phi$ as an integral over the wedge and its opening angle,
by substituting $s$ with the position on the wedge $\ell \in [1,L]$ and the angle $\phi \in [0, \Phi]$:

$$
b_{i,j} = \frac{1}{\lvert C \rvert} \int_{\ell} \int_{\phi} \lvert h\left(s(\ell,\phi) \right) \rvert^2 \mathbb{1}_{\left\{ s(\ell,\phi) \in C_{i,j} \right\}} \left\lVert \frac{\partial r}{\partial \ell} \times \frac{\partial r}{\partial \phi} \right\rVert d\ell d\phi
$$
    
where the integral is over the wedge length $L$ and opening angle $\Phi$, and
$r\left( \ell, \phi \right)$ is the reparametrization with respected to $(\ell, \phi)$ of the
intersection between the diffraction cone at $\ell$ and the rectangle defining the coverage map (see, e.g., <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#surfaceintegral" id="id2">[SurfaceIntegral]</a>).
The previous integral is approximated through Monte Carlo sampling by shooting $N'$ rays from equally spaced
locations $\ell_n$ along the wedge with directions $\phi_n$ sampled uniformly from $(0, \Phi)$:

$$
\hat{b}_{i,j}^{\text{(diff)}} = \frac{L\Phi}{N'\lvert C \rvert} \sum_{n=1}^{N'} \lvert h\left(s(\ell_n,\phi_n)\right) \rvert^2 \mathbb{1}_{\left\{ s(\ell_n,\phi_n) \in C_{i,j} \right\}} \left\lVert \left(\frac{\partial r}{\partial \ell}\right)_n \times \left(\frac{\partial r}{\partial \phi}\right)_n \right\rVert.
$$
    
The output of this function is therefore a real-valued matrix of size `[num_cells_y,` `num_cells_x]`,
for every transmitter, with elements equal to the sum of the contributions of the reflected and scattered paths
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#equation-cm-mc-ref">(44)</a> and diffracted paths <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#equation-cm-mc-diff">(45)</a> for all the wedges, and where

$$
\begin{split}\texttt{num_cells_x} = \bigg\lceil\frac{\texttt{cm_size[0]}}{\texttt{cm_cell_size[0]}} \bigg\rceil\\
\texttt{num_cells_y} = \bigg\lceil \frac{\texttt{cm_size[1]}}{\texttt{cm_cell_size[1]}} \bigg\rceil.\end{split}
$$
    
The surface defining the coverage map is a rectangle centered at
`cm_center`, with orientation `cm_orientation`, and with size
`cm_size`. An orientation of (0,0,0) corresponds to
a coverage map parallel to the XY plane, with surface normal pointing towards
the $+z$ axis. By default, the coverage map
is parallel to the XY plane, covers all of the scene, and has
an elevation of $z = 1.5\text{m}$.
The receiver is assumed to use the antenna array
`scene.rx_array`. If transmitter and/or receiver have multiple antennas, transmit precoding
and receive combining are applied which are defined by `precoding_vec` and
`combining_vec`, respectively.
    
The $(i,j)$ indices are omitted in the following for clarity.
For reflection and scattering, paths are generated by shooting `num_samples` rays from the
transmitters with directions arranged in a Fibonacci lattice on the unit
sphere and by simulating their propagation for up to `max_depth` interactions with
scene objects.
If `max_depth` is set to 0 and if `los` is set to <cite>True</cite>,
only the line-of-sight path is considered.
For diffraction, paths are generated by shooting `num_samples` rays from equally
spaced locations along the wedges in line-of-sight with the transmitter, with
directions uniformly sampled on the diffraction cone.
    
For every ray $n$ intersecting the coverage map cell $(i,j)$, the
channel coefficients, $a_n$, and the angles of departure (AoDs)
$(\theta_{\text{T},n}, \varphi_{\text{T},n})$
and arrival (AoAs) $(\theta_{\text{R},n}, \varphi_{\text{R},n})$
are computed. See the <a class="reference external" href="../em_primer.html">Primer on Electromagnetics</a> for more details.
    
A “synthetic” array is simulated by adding additional phase shifts that depend on the
antenna position relative to the position of the transmitter (receiver) as well as the AoDs (AoAs).
For the $k^\text{th}$ transmit antenna and $\ell^\text{th}$ receive antenna, let
us denote by $\mathbf{d}_{\text{T},k}$ and $\mathbf{d}_{\text{R},\ell}$ the relative positions (with respect to
the positions of the transmitter/receiver) of the pair of antennas
for which the channel impulse response shall be computed. These can be accessed through the antenna array’s property
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.AntennaArray.positions" title="sionna.rt.AntennaArray.positions">`positions`</a>. Using a plane-wave assumption, the resulting phase shifts
from these displacements can be computed as

$$
\begin{split}p_{\text{T}, n,k} &= \frac{2\pi}{\lambda}\hat{\mathbf{r}}(\theta_{\text{T},n}, \varphi_{\text{T},n})^\mathsf{T} \mathbf{d}_{\text{T},k}\\
p_{\text{R}, n,\ell} &= \frac{2\pi}{\lambda}\hat{\mathbf{r}}(\theta_{\text{R},n}, \varphi_{\text{R},n})^\mathsf{T} \mathbf{d}_{\text{R},\ell}.\end{split}
$$
    
The final expression for the path coefficient is

$$
h_{n,k,\ell} =  a_n e^{j(p_{\text{T}, i,k} + p_{\text{R}, i,\ell})}
$$
    
for every transmit antenna $k$ and receive antenna $\ell$.
These coefficients form the complex-valued channel matrix, $\mathbf{H}_n$,
of size $\texttt{num_rx_ant} \times \texttt{num_tx_ant}$.
    
Finally, the coefficient of the equivalent SISO channel is

$$
h_n =  \mathbf{c}^{\mathsf{H}} \mathbf{H}_n \mathbf{p}
$$
    
where $\mathbf{c}$ and $\mathbf{p}$ are the combining and
precoding vectors (`combining_vec` and `precoding_vec`),
respectively.
<p class="rubric">Example
```python
import sionna
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver
scene = load_scene(sionna.rt.scene.munich)
# Configure antenna array for all transmitters
scene.tx_array = PlanarArray(num_rows=8,
                        num_cols=2,
                        vertical_spacing=0.7,
                        horizontal_spacing=0.5,
                        pattern="tr38901",
                        polarization="VH")
# Configure antenna array for all receivers
scene.rx_array = PlanarArray(num_rows=1,
                        num_cols=1,
                        vertical_spacing=0.5,
                        horizontal_spacing=0.5,
                        pattern="dipole",
                        polarization="cross")
# Add a transmitters
tx = Transmitter(name="tx",
            position=[8.5,21,30],
            orientation=[0,0,0])
scene.add(tx)
tx.look_at([40,80,1.5])
# Compute coverage map
cm = scene.coverage_map(cm_cell_size=[1.,1.],
                    num_samples=int(10e6))
# Visualize coverage in preview
scene.preview(coverage_map=cm,
            resolution=[1000, 600])
```

<img alt="../_images/coverage_map_preview.png" src="https://nvlabs.github.io/sionna/_images/coverage_map_preview.png" />

Input
 
- **rx_orientation** (<em>[3], float</em>) – Orientation of the receiver $(\alpha, \beta, \gamma)$
specified through three angles corresponding to a 3D rotation
as defined in <a class="reference internal" href="../em_primer.html#equation-rotation">(3)</a>. Defaults to $(0,0,0)$.
- **max_depth** (<em>int</em>) – Maximum depth (i.e., number of bounces) allowed for tracing the
paths. Defaults to 3.
- **cm_center** ([3], float | <cite>None</cite>) – Center of the coverage map $(x,y,z)$ as three-dimensional
vector. If set to <cite>None</cite>, the coverage map is centered on the
center of the scene, except for the elevation $z$ that is set
to 1.5m. Otherwise, `cm_orientation` and `cm_scale` must also
not be <cite>None</cite>. Defaults to <cite>None</cite>.
- **cm_orientation** ([3], float | <cite>None</cite>) – Orientation of the coverage map $(\alpha, \beta, \gamma)$
specified through three angles corresponding to a 3D rotation
as defined in <a class="reference internal" href="../em_primer.html#equation-rotation">(3)</a>.
An orientation of $(0,0,0)$ or <cite>None</cite> corresponds to a
coverage map that is parallel to the XY plane.
If not set to <cite>None</cite>, then `cm_center` and `cm_scale` must also
not be <cite>None</cite>.
Defaults to <cite>None</cite>.
- **cm_size** ([2], float | <cite>None</cite>) – Size of the coverage map [m].
If set to <cite>None</cite>, then the size of the coverage map is set such that
it covers the entire scene.
Otherwise, `cm_center` and `cm_orientation` must also not be
<cite>None</cite>. Defaults to <cite>None</cite>.
- **cm_cell_size** (<em>[2], float</em>) – Size of a cell of the coverage map [m].
Defaults to $(10,10)$.
- **combining_vec** (<em>[num_rx_ant], complex | None</em>) – Combining vector.
If set to <cite>None</cite>, then no combining is applied, and
the energy received by all antennas is summed.
- **precoding_vec** (<em>[num_tx_ant], complex | None</em>) – Precoding vector.
If set to <cite>None</cite>, then defaults to
$\frac{1}{\sqrt{\text{num_tx_ant}}} [1,\dots,1]^{\mathsf{T}}$.
- **num_samples** (<em>int</em>) – Number of random rays to trace.
For the reflected paths, this number is split equally over the different transmitters.
For the diffracted paths, it is split over the wedges in line-of-sight with the
transmitters such that the number of rays allocated
to a wedge is proportional to its length.
Defaults to 2e6.
- **los** (<em>bool</em>) – If set to <cite>True</cite>, then the LoS paths are computed.
Defaults to <cite>True</cite>.
- **reflection** (<em>bool</em>) – If set to <cite>True</cite>, then the reflected paths are computed.
Defaults to <cite>True</cite>.
- **diffraction** (<em>bool</em>) – If set to <cite>True</cite>, then the diffracted paths are computed.
Defaults to <cite>False</cite>.
- **scattering** (<em>bool</em>) – If set to <cite>True</cite>, then the scattered paths are computed.
Defaults to <cite>False</cite>.
- **edge_diffraction** (<em>bool</em>) – If set to <cite>False</cite>, only diffraction on wedges, i.e., edges that
connect two primitives, is considered.
Defaults to <cite>False</cite>.
- **check_scene** (<em>bool</em>) – If set to <cite>True</cite>, checks that the scene is well configured before
computing the coverage map. This can add a significant overhead.
Defaults to <cite>True</cite>.


Output
    
cm : <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap" title="sionna.rt.CoverageMap">`CoverageMap`</a> – The coverage maps



### load_scene<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#load-scene" title="Permalink to this headline"></a>

`sionna.rt.``load_scene`(<em class="sig-param">`filename``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/rt/scene.html#load_scene">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.load_scene" title="Permalink to this definition"></a>
    
Load a scene from file
    
Note that only one scene can be loaded at a time.
Input
 
- **filename** (<em>str</em>) – Name of a valid scene file. Sionna uses the simple XML-based format
from <a class="reference external" href="https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html">Mitsuba 3</a>.
Defaults to <cite>None</cite> for which an empty scene is created.
- **dtype** (<em>tf.complex</em>) – Dtype used for all internal computations and outputs.
Defaults to <cite>tf.complex64</cite>.


Output
    
**scene** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene" title="sionna.rt.Scene">`Scene`</a>) – Reference to the current scene



### preview<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#preview" title="Permalink to this headline"></a>

`sionna.rt.Scene.``preview`(<em class="sig-param">`paths``=``None`</em>, <em class="sig-param">`show_paths``=``True`</em>, <em class="sig-param">`show_devices``=``True`</em>, <em class="sig-param">`coverage_map``=``None`</em>, <em class="sig-param">`cm_tx``=``0`</em>, <em class="sig-param">`cm_vmin``=``None`</em>, <em class="sig-param">`cm_vmax``=``None`</em>, <em class="sig-param">`resolution``=``(655,` `500)`</em>, <em class="sig-param">`fov``=``45`</em>, <em class="sig-param">`background``=``'#ffffff'`</em>, <em class="sig-param">`clip_at``=``None`</em>, <em class="sig-param">`clip_plane_orientation``=``(0,` `0,` `-` `1)`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.preview" title="Permalink to this definition"></a>
    
In an interactive notebook environment, opens an interactive 3D
viewer of the scene.
    
The returned value of this method must be the last line of
the cell so that it is displayed. For example:
```python
fig = scene.preview()
# ...
fig
```

    
Or simply:
```python
scene.preview()
```

    
Color coding:
 
- Green: Receiver
- Blue: Transmitter

    
Controls:
 
- Mouse left: Rotate
- Scroll wheel: Zoom
- Mouse right: Move

Input
 
- **paths** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a> | <cite>None</cite>) – Simulated paths generated by
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.compute_paths" title="sionna.rt.Scene.compute_paths">`compute_paths()`</a> or <cite>None</cite>.
If <cite>None</cite>, only the scene is rendered.
Defaults to <cite>None</cite>.
- **show_paths** (<em>bool</em>) – If <cite>paths</cite> is not <cite>None</cite>, shows the paths.
Defaults to <cite>True</cite>.
- **show_devices** (<em>bool</em>) – If set to <cite>True</cite>, shows the radio devices.
Defaults to <cite>True</cite>.
- **show_orientations** (<em>bool</em>) – If <cite>show_devices</cite> is <cite>True</cite>, shows the radio devices orientations.
Defaults to <cite>False</cite>.
- **coverage_map** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap" title="sionna.rt.CoverageMap">`CoverageMap`</a> | <cite>None</cite>) – An optional coverage map to overlay in the scene for visualization.
Defaults to <cite>None</cite>.
- **cm_tx** (<em>int | str</em>) – When <cite>coverage_map</cite> is specified, controls which of the transmitters
to display the coverage map for. Either the transmitter’s name
or index can be given.
Defaults to <cite>0</cite>.
- **cm_db_scale** (<em>bool</em>) – Use logarithmic scale for coverage map visualization, i.e. the
coverage values are mapped with:
$y = 10 \cdot \log_{10}(x)$.
Defaults to <cite>True</cite>.
- **cm_vmin, cm_vmax** (<em>floot | None</em>) – For coverage map visualization, defines the range of path gains that
the colormap covers.
These parameters should be provided in dB if `cm_db_scale` is
set to <cite>True</cite>, or in linear scale otherwise.
If set to None, then covers the complete range.
Defaults to <cite>None</cite>.
- **resolution** (<em>[2], int</em>) – Size of the viewer figure.
Defaults to <cite>[655, 500]</cite>.
- **fov** (<em>float</em>) – Field of view, in degrees.
Defaults to 45°.
- **background** (<em>str</em>) – Background color in hex format prefixed by ‘#’.
Defaults to ‘#ffffff’ (white).
- **clip_at** (<em>float</em>) – If not <cite>None</cite>, the scene preview will be clipped (cut) by a plane
with normal orientation `clip_plane_orientation` and offset `clip_at`.
That means that everything <em>behind</em> the plane becomes invisible.
This allows visualizing the interior of meshes, such as buildings.
Defaults to <cite>None</cite>.
- **clip_plane_orientation** (<em>tuple[float, float, float]</em>) – Normal vector of the clipping plane.
Defaults to (0,0,-1).




### render<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#render" title="Permalink to this headline"></a>

`sionna.rt.Scene.``render`(<em class="sig-param">`camera`</em>, <em class="sig-param">`paths``=``None`</em>, <em class="sig-param">`show_paths``=``True`</em>, <em class="sig-param">`show_devices``=``True`</em>, <em class="sig-param">`coverage_map``=``None`</em>, <em class="sig-param">`cm_tx``=``0`</em>, <em class="sig-param">`cm_vmin``=``None`</em>, <em class="sig-param">`cm_vmax``=``None`</em>, <em class="sig-param">`cm_show_color_bar``=``True`</em>, <em class="sig-param">`num_samples``=``512`</em>, <em class="sig-param">`resolution``=``(655,` `500)`</em>, <em class="sig-param">`fov``=``45`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.render" title="Permalink to this definition"></a>
    
Renders the scene from the viewpoint of a camera or the interactive
viewer
Input
 
- **camera** (str | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a>) – The name or instance of a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a>.
If an interactive viewer was opened with
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.preview" title="sionna.rt.Scene.preview">`preview()`</a>, set to <cite>“preview”</cite> to use its
viewpoint.
- **paths** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a> | <cite>None</cite>) – Simulated paths generated by
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.compute_paths" title="sionna.rt.Scene.compute_paths">`compute_paths()`</a> or <cite>None</cite>.
If <cite>None</cite>, only the scene is rendered.
Defaults to <cite>None</cite>.
- **show_paths** (<em>bool</em>) – If <cite>paths</cite> is not <cite>None</cite>, shows the paths.
Defaults to <cite>True</cite>.
- **show_devices** (<em>bool</em>) – If <cite>paths</cite> is not <cite>None</cite>, shows the radio devices.
Defaults to <cite>True</cite>.
- **coverage_map** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap" title="sionna.rt.CoverageMap">`CoverageMap`</a> | <cite>None</cite>) – An optional coverage map to overlay in the scene for visualization.
Defaults to <cite>None</cite>.
- **cm_tx** (<em>int | str</em>) – When <cite>coverage_map</cite> is specified, controls which of the transmitters
to display the coverage map for. Either the transmitter’s name
or index can be given.
Defaults to <cite>0</cite>.
- **cm_db_scale** (<em>bool</em>) – Use logarithmic scale for coverage map visualization, i.e. the
coverage values are mapped with:
$y = 10 \cdot \log_{10}(x)$.
Defaults to <cite>True</cite>.
- **cm_vmin, cm_vmax** (<em>float | None</em>) – For coverage map visualization, defines the range of path gains that
the colormap covers.
These parameters should be provided in dB if `cm_db_scale` is
set to <cite>True</cite>, or in linear scale otherwise.
If set to None, then covers the complete range.
Defaults to <cite>None</cite>.
- **cm_show_color_bar** (<em>bool</em>) – For coverage map visualization, show the color bar describing the
color mapping used next to the rendering.
Defaults to <cite>True</cite>.
- **num_samples** (<em>int</em>) – Number of rays thrown per pixel.
Defaults to 512.
- **resolution** (<em>[2], int</em>) – Size of the rendered figure.
Defaults to <cite>[655, 500]</cite>.
- **fov** (<em>float</em>) – Field of view, in degrees.
Defaults to 45°.


Output
    
`Figure` – Rendered image



### render_to_file<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#render-to-file" title="Permalink to this headline"></a>

`sionna.rt.Scene.``render_to_file`(<em class="sig-param">`camera`</em>, <em class="sig-param">`filename`</em>, <em class="sig-param">`paths``=``None`</em>, <em class="sig-param">`show_paths``=``True`</em>, <em class="sig-param">`show_devices``=``True`</em>, <em class="sig-param">`coverage_map``=``None`</em>, <em class="sig-param">`cm_tx``=``0`</em>, <em class="sig-param">`cm_db_scale``=``True`</em>, <em class="sig-param">`cm_vmin``=``None`</em>, <em class="sig-param">`cm_vmax``=``None`</em>, <em class="sig-param">`num_samples``=``512`</em>, <em class="sig-param">`resolution``=``(655,` `500)`</em>, <em class="sig-param">`fov``=``45`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.render_to_file" title="Permalink to this definition"></a>
    
Renders the scene from the viewpoint of a camera or the interactive
viewer, and saves the resulting image
Input
 
- **camera** (str | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a>) – The name or instance of a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a>.
If an interactive viewer was opened with
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.preview" title="sionna.rt.Scene.preview">`preview()`</a>, set to <cite>“preview”</cite> to use its
viewpoint.
- **filename** (<em>str</em>) – Filename for saving the rendered image, e.g., “my_scene.png”
- **paths** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a> | <cite>None</cite>) – Simulated paths generated by
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.compute_paths" title="sionna.rt.Scene.compute_paths">`compute_paths()`</a> or <cite>None</cite>.
If <cite>None</cite>, only the scene is rendered.
Defaults to <cite>None</cite>.
- **show_paths** (<em>bool</em>) – If <cite>paths</cite> is not <cite>None</cite>, shows the paths.
Defaults to <cite>True</cite>.
- **show_devices** (<em>bool</em>) – If <cite>paths</cite> is not <cite>None</cite>, shows the radio devices.
Defaults to <cite>True</cite>.
- **coverage_map** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap" title="sionna.rt.CoverageMap">`CoverageMap`</a> | <cite>None</cite>) – An optional coverage map to overlay in the scene for visualization.
Defaults to <cite>None</cite>.
- **cm_tx** (<em>int | str</em>) – When <cite>coverage_map</cite> is specified, controls which of the transmitters
to display the coverage map for. Either the transmitter’s name
or index can be given.
Defaults to <cite>0</cite>.
- **cm_db_scale** (<em>bool</em>) – Use logarithmic scale for coverage map visualization, i.e. the
coverage values are mapped with:
$y = 10 \cdot \log_{10}(x)$.
Defaults to <cite>True</cite>.
- **cm_vmin, cm_vmax** (<em>float | None</em>) – For coverage map visualization, defines the range of path gains that
the colormap covers.
These parameters should be provided in dB if `cm_db_scale` is
set to <cite>True</cite>, or in linear scale otherwise.
If set to None, then covers the complete range.
Defaults to <cite>None</cite>.
- **num_samples** (<em>int</em>) – Number of rays thrown per pixel.
Defaults to 512.
- **resolution** (<em>[2], int</em>) – Size of the rendered figure.
Defaults to <cite>[655, 500]</cite>.
- **fov** (<em>float</em>) – Field of view, in degrees.
Defaults to 45°.




## Example Scenes<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#example-scenes" title="Permalink to this headline"></a>
    
Sionna has several integrated scenes that are listed below.
They can be loaded and used as follows:
```python
scene = load_scene(sionna.rt.scene.etoile)
scene.preview()
```
### floor_wall<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#floor-wall" title="Permalink to this headline"></a>

`sionna.rt.scene.``floor_wall`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.scene.floor_wall" title="Permalink to this definition"></a>
    
Example scene containing a ground plane and a vertical wall
<img alt="../_images/floor_wall.png" src="https://nvlabs.github.io/sionna/_images/floor_wall.png" />

    
(<a class="reference external" href="https://drive.google.com/file/d/1djXBj3VYLT4_bQpmp4vR6o6agGmv_p1F/view?usp=share_link">Blender file</a>)

### simple_street_canyon<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#simple-street-canyon" title="Permalink to this headline"></a>

`sionna.rt.scene.``simple_street_canyon`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.scene.simple_street_canyon" title="Permalink to this definition"></a>
    
Example scene containing a few rectangular building blocks and a ground plane
<img alt="../_images/street_canyon.png" src="https://nvlabs.github.io/sionna/_images/street_canyon.png" />

    
(<a class="reference external" href="https://drive.google.com/file/d/1_1nsLtSC8cy1QfRHAN_JetT3rPP21tNb/view?usp=share_link">Blender file</a>)

### etoile<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#etoile" title="Permalink to this headline"></a>

`sionna.rt.scene.``etoile`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.scene.etoile" title="Permalink to this definition"></a>
    
Example scene containing the area around the Arc de Triomphe in Paris
The scene was created with data downloaded from <a class="reference external" href="https://www.openstreetmap.org">OpenStreetMap</a> and
the help of <a class="reference external" href="https://www.blender.org">Blender</a> and the <a class="reference external" href="https://github.com/vvoovv/blender-osm">Blender-OSM</a>
and <a class="reference external" href="https://github.com/mitsuba-renderer/mitsuba-blender">Mitsuba Blender</a> add-ons.
The data is licensed under the <a class="reference external" href="https://openstreetmap.org/copyright">Open Data Commons Open Database License (ODbL)</a>.
<img alt="../_images/etoile.png" src="https://nvlabs.github.io/sionna/_images/etoile.png" />

    
(<a class="reference external" href="https://drive.google.com/file/d/1bamQ67lLGZHTfNmcVajQDmq2oiSY8FEn/view?usp=share_link">Blender file</a>)

### munich<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#munich" title="Permalink to this headline"></a>

`sionna.rt.scene.``munich`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.scene.munich" title="Permalink to this definition"></a>
    
Example scene containing the area around the Frauenkirche in Munich
The scene was created with data downloaded from <a class="reference external" href="https://www.openstreetmap.org">OpenStreetMap</a> and
the help of <a class="reference external" href="https://www.blender.org">Blender</a> and the <a class="reference external" href="https://github.com/vvoovv/blender-osm">Blender-OSM</a>
and <a class="reference external" href="https://github.com/mitsuba-renderer/mitsuba-blender">Mitsuba Blender</a> add-ons.
The data is licensed under the <a class="reference external" href="https://openstreetmap.org/copyright">Open Data Commons Open Database License (ODbL)</a>.
<img alt="../_images/munich.png" src="https://nvlabs.github.io/sionna/_images/munich.png" />

    
(<a class="reference external" href="https://drive.google.com/file/d/15WrvMGrPWsoVKYvDG6Ab7btq-ktTCGR1/view?usp=share_link">Blender file</a>)

### simple_wedge<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#simple-wedge" title="Permalink to this headline"></a>

`sionna.rt.scene.``simple_wedge`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.scene.simple_wedge" title="Permalink to this definition"></a>
    
Example scene containing a wedge with a $90^{\circ}$ opening angle
<img alt="../_images/simple_wedge.png" src="https://nvlabs.github.io/sionna/_images/simple_wedge.png" />

    
(<a class="reference external" href="https://drive.google.com/file/d/1RnJoYzXKkILMEmf-UVSsyjq-EowU6JRA/view?usp=share_link">Blender file</a>)

### simple_reflector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#simple-reflector" title="Permalink to this headline"></a>

`sionna.rt.scene.``simple_reflector`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.scene.simple_reflector" title="Permalink to this definition"></a>
    
Example scene containing a metallic square
<img alt="../_images/simple_reflector.png" src="https://nvlabs.github.io/sionna/_images/simple_reflector.png" />

    
(<a class="reference external" href="https://drive.google.com/file/d/1iYPD11zAAMj0gNUKv_nv6QdLhOJcPpIa/view?usp=share_link">Blender file</a>)

### double_reflector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#double-reflector" title="Permalink to this headline"></a>

`sionna.rt.scene.``double_reflector`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.scene.double_reflector" title="Permalink to this definition"></a>
    
Example scene containing two metallic squares
<img alt="../_images/double_reflector.png" src="https://nvlabs.github.io/sionna/_images/double_reflector.png" />

    
(<a class="reference external" href="https://drive.google.com/file/d/1K2ZUYHPPkrq9iUauJtInRu7x2r16D1zN/view?usp=share_link">Blender file</a>)

### triple_reflector<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#triple-reflector" title="Permalink to this headline"></a>

`sionna.rt.scene.``triple_reflector`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.scene.triple_reflector" title="Permalink to this definition"></a>
    
Example scene containing three metallic rectangles
<img alt="../_images/triple_reflector.png" src="https://nvlabs.github.io/sionna/_images/triple_reflector.png" />

    
(<a class="reference external" href="https://drive.google.com/file/d/1l95_0U2b3cEVtz3G8mQxuLxy8xiPsVID/view?usp=share_link">Blender file</a>)

### Box<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#box" title="Permalink to this headline"></a>

`sionna.rt.scene.``box`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.scene.box" title="Permalink to this definition"></a>
    
Example scene containing a metallic box
<img alt="../_images/box.png" src="https://nvlabs.github.io/sionna/_images/box.png" />

    
(<a class="reference external" href="https://drive.google.com/file/d/1pywetyKr0HBz3aSYpkmykGnjs_1JMsHY/view?usp=share_link">Blender file</a>)
## Paths<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#paths" title="Permalink to this headline"></a>
    
A propagation path $i$ starts at a transmit antenna and ends at a receive antenna. It is described by
its channel coefficient $a_i$ and delay $\tau_i$, as well as the
angles of departure $(\theta_{\text{T},i}, \varphi_{\text{T},i})$
and arrival $(\theta_{\text{R},i}, \varphi_{\text{R},i})$.
For more detail, see the <a class="reference external" href="../em_primer.html">Primer on Electromagnetics</a>.
    
In Sionna, paths are computed with the help of the function <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.compute_paths" title="sionna.rt.Scene.compute_paths">`compute_paths()`</a> which returns an instance of
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="sionna.rt.Paths">`Paths`</a>. Paths can be visualized by providing them as arguments to the functions <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.render" title="sionna.rt.Scene.render">`render()`</a>,
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.render_to_file" title="sionna.rt.Scene.render_to_file">`render_to_file()`</a>, or <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.preview" title="sionna.rt.Scene.preview">`preview()`</a>.
    
Channel impulse responses (CIRs) can be obtained with <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.cir" title="sionna.rt.Paths.cir">`cir()`</a> which can
then be used for link-level simulations. This is for example done in the <a class="reference external" href="../examples/Sionna_Ray_Tracing_Introduction.html">Sionna Ray Tracing Tutorial</a>.

### Paths<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#id13" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.rt.``Paths`<a class="reference internal" href="../_modules/sionna/rt/paths.html#Paths">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths" title="Permalink to this definition"></a>
    
Stores the simulated propagation paths
    
Paths are generated for the loaded scene using
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.compute_paths" title="sionna.rt.Scene.compute_paths">`compute_paths()`</a>. Please refer to the
documentation of this function for further details.
These paths can then be used to compute channel impulse responses:
```python
paths = scene.compute_paths()
a, tau = paths.cir()
```

    
where `scene` is the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene" title="sionna.rt.Scene">`Scene`</a> loaded using
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.load_scene" title="sionna.rt.load_scene">`load_scene()`</a>.

<em class="property">`property` </em>`a`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.a" title="Permalink to this definition"></a>
    
Passband channel coefficients $a_i$ of each path as defined in <a class="reference internal" href="../em_primer.html#equation-h-final">(26)</a>.
Type
    
[batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, num_time_steps], tf.complex




`apply_doppler`(<em class="sig-param">`sampling_frequency`</em>, <em class="sig-param">`num_time_steps`</em>, <em class="sig-param">`tx_velocities``=``(0.0,` `0.0,` `0.0)`</em>, <em class="sig-param">`rx_velocities``=``(0.0,` `0.0,` `0.0)`</em>)<a class="reference internal" href="../_modules/sionna/rt/paths.html#Paths.apply_doppler">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.apply_doppler" title="Permalink to this definition"></a>
    
Apply Doppler shifts corresponding to input transmitters and receivers
velocities.
    
This function replaces the last dimension of the tensor storing the
paths coefficients <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.a" title="sionna.rt.Paths.a">`a`</a>, which stores the the temporal evolution of
the channel, with a dimension of size `num_time_steps` computed
according to the input velocities.
    
Time evolution of the channel coefficients is simulated by computing the
Doppler shift due to movements of the transmitter and receiver. If we denote by
$\mathbf{v}_{\text{T}}\in\mathbb{R}^3$ and $\mathbf{v}_{\text{R}}\in\mathbb{R}^3$
the velocity vectors of the transmitter and receiver, respectively, the Doppler shifts are computed as

$$
\begin{split}f_{\text{T}, i} &= \frac{\hat{\mathbf{r}}(\theta_{\text{T},i}, \varphi_{\text{T},i})^\mathsf{T}\mathbf{v}_{\text{T}}}{\lambda}\qquad \text{[Hz]}\\
f_{\text{R}, i} &= \frac{\hat{\mathbf{r}}(\theta_{\text{R},i}, \varphi_{\text{R},i})^\mathsf{T}\mathbf{v}_{\text{R}}}{\lambda}\qquad \text{[Hz]}\end{split}
$$
    
for an arbitrary path $i$, where $(\theta_{\text{T},i}, \varphi_{\text{T},i})$ are the AoDs,
$(\theta_{\text{R},i}, \varphi_{\text{R},i})$ are the AoAs, and $\lambda$ is the wavelength.
This leads to the time-dependent path coefficient

$$
a_i(t) = a_i e^{j2\pi(f_{\text{T}, i}+f_{\text{R}, i})t}.
$$
    
Note that this model is only valid as long as the AoDs, AoAs, and path delay do not change.
    
When this function is called multiple times, it overwrites the previous
time steps dimension.
Input
 
- **sampling_frequency** (<em>float</em>) – Frequency [Hz] at which the channel impulse response is sampled
- **num_time_steps** (<em>int</em>) – Number of time steps.
- **tx_velocities** ([batch_size, num_tx, 3] or broadcastable, tf.float | <cite>None</cite>) – Velocity vectors $(v_\text{x}, v_\text{y}, v_\text{z})$ of all
transmitters [m/s].
Defaults to <cite>[0,0,0]</cite>.
- **rx_velocities** ([batch_size, num_tx, 3] or broadcastable, tf.float | <cite>None</cite>) – Velocity vectors $(v_\text{x}, v_\text{y}, v_\text{z})$ of all
receivers [m/s].
Defaults to <cite>[0,0,0]</cite>.





`cir`(<em class="sig-param">`los``=``True`</em>, <em class="sig-param">`reflection``=``True`</em>, <em class="sig-param">`diffraction``=``True`</em>, <em class="sig-param">`scattering``=``True`</em>, <em class="sig-param">`num_paths``=``None`</em>)<a class="reference internal" href="../_modules/sionna/rt/paths.html#Paths.cir">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.cir" title="Permalink to this definition"></a>
    
Returns the baseband equivalent channel impulse response <a class="reference internal" href="../em_primer.html#equation-h-b">(28)</a>
which can be used for link simulations by other Sionna components.
    
The baseband equivalent channel coefficients $a^{\text{b}}_{i}$
are computed as :

$$
a^{\text{b}}_{i} = a_{i} e^{-j2 \pi f \tau_{i}}
$$
    
where $i$ is the index of an arbitrary path, $a_{i}$
is the passband path coefficient (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.a" title="sionna.rt.Paths.a">`a`</a>),
$\tau_{i}$ is the path delay (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.tau" title="sionna.rt.Paths.tau">`tau`</a>),
and $f$ is the carrier frequency.
    
Note: For the paths of a given type to be returned (LoS, reflection, etc.), they
must have been previously computed by <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.compute_paths" title="sionna.rt.Scene.compute_paths">`compute_paths()`</a>, i.e.,
the corresponding flags must have been set to <cite>True</cite>.
Input
 
- **los** (<em>bool</em>) – If set to <cite>False</cite>, LoS paths are not returned.
Defaults to <cite>True</cite>.
- **reflection** (<em>bool</em>) – If set to <cite>False</cite>, specular paths are not returned.
Defaults to <cite>True</cite>.
- **diffraction** (<em>bool</em>) – If set to <cite>False</cite>, diffracted paths are not returned.
Defaults to <cite>True</cite>.
- **scattering** (<em>bool</em>) – If set to <cite>False</cite>, scattered paths are not returned.
Defaults to <cite>True</cite>.
- **num_paths** (int or <cite>None</cite>) – All CIRs are either zero-padded or cropped to the largest
`num_paths` paths.
Defaults to <cite>None</cite> which means that no padding or cropping is done.


Output
 
- **a** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, num_time_steps], tf.complex</em>) – Path coefficients
- **tau** (<em>[batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float</em>) – Path delays





`export`(<em class="sig-param">`filename`</em>)<a class="reference internal" href="../_modules/sionna/rt/paths.html#Paths.export">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.export" title="Permalink to this definition"></a>
    
Saves the paths as an OBJ file for visualisation, e.g., in Blender
Input
    
**filename** (<em>str</em>) – Path and name of the file




`from_dict`(<em class="sig-param">`data_dict`</em>)<a class="reference internal" href="../_modules/sionna/rt/paths.html#Paths.from_dict">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.from_dict" title="Permalink to this definition"></a>
    
Set the paths from a dictionnary which values are tensors
    
The format of the dictionnary is expected to be the same as the one
returned by <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.to_dict" title="sionna.rt.Paths.to_dict">`to_dict()`</a>.
Input
    
**data_dict** (<cite>dict</cite>)




<em class="property">`property` </em>`mask`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.mask" title="Permalink to this definition"></a>
    
Set to <cite>False</cite> for non-existent paths.
When there are multiple transmitters or receivers, path counts may vary between links. This is used to identify non-existent paths.
For such paths, the channel coefficient is set to <cite>0</cite> and the delay to <cite>-1</cite>.
Type
    
[batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.bool




<em class="property">`property` </em>`normalize_delays`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.normalize_delays" title="Permalink to this definition"></a>
    
Set to <cite>True</cite> to normalize path delays such that the first path
between any pair of antennas of a transmitter and receiver arrives at
`tau` `=` `0`. Defaults to <cite>True</cite>.
Type
    
bool




<em class="property">`property` </em>`phi_r`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.phi_r" title="Permalink to this definition"></a>
    
Azimuth angles of arrival [rad]
Type
    
[batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float




<em class="property">`property` </em>`phi_t`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.phi_t" title="Permalink to this definition"></a>
    
Azimuth angles of departure [rad]
Type
    
[batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float




<em class="property">`property` </em>`reverse_direction`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.reverse_direction" title="Permalink to this definition"></a>
    
If set to <cite>True</cite>, swaps receivers and transmitters
Type
    
bool




<em class="property">`property` </em>`tau`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.tau" title="Permalink to this definition"></a>
    
Propagation delay $\tau_i$ [s] of each path as defined in <a class="reference internal" href="../em_primer.html#equation-h-final">(26)</a>.
Type
    
[batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float




<em class="property">`property` </em>`theta_r`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.theta_r" title="Permalink to this definition"></a>
    
Zenith angles of arrival [rad]
Type
    
[batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float




<em class="property">`property` </em>`theta_t`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.theta_t" title="Permalink to this definition"></a>
    
Zenith  angles of departure [rad]
Type
    
[batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float




`to_dict`()<a class="reference internal" href="../_modules/sionna/rt/paths.html#Paths.to_dict">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.to_dict" title="Permalink to this definition"></a>
    
Returns the properties of the paths as a dictionnary which values are
tensors
Output
    
<cite>dict</cite>




<em class="property">`property` </em>`types`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Paths.types" title="Permalink to this definition"></a>
    
Type of the paths:
 
- 0 : LoS
- 1 : Reflected
- 2 : Diffracted
- 3 : Scattered

Type
    
[batch_size, max_num_paths], tf.int




## Coverage Maps<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#coverage-maps" title="Permalink to this headline"></a>
    
A coverage map describes the received power from a specific transmitter at every point on a plane.
In other words, for a given transmitter, it associates every point on a surface  with the power that a receiver with
a specific orientation would observe at this point. A coverage map is not uniquely defined as it depends on
the transmit and receive arrays and their respective antenna patterns, the transmitter and receiver orientations, as well as
transmit precoding and receive combining vectors. Moreover, a coverage map is not continuous but discrete because the plane
needs to be quantized into small rectangular bins.
    
In Sionna, coverage maps are computed with the help of the function <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.coverage_map" title="sionna.rt.Scene.coverage_map">`coverage_map()`</a> which returns an instance of
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap" title="sionna.rt.CoverageMap">`CoverageMap`</a>. They can be visualized by providing them either as arguments to the functions <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.render" title="sionna.rt.Scene.render">`render()`</a>,
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.render_to_file" title="sionna.rt.Scene.render_to_file">`render_to_file()`</a>, and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.preview" title="sionna.rt.Scene.preview">`preview()`</a>, or by using the class method <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap.show" title="sionna.rt.CoverageMap.show">`show()`</a>.
    
A very useful feature is <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap.sample_positions" title="sionna.rt.CoverageMap.sample_positions">`sample_positions()`</a> which allows sampling
of random positions within the scene that have sufficient coverage from a specific transmitter.
This feature is used in the <a class="reference external" href="../examples/Sionna_Ray_Tracing_Introduction.html">Sionna Ray Tracing Tutorial</a> to generate a dataset of channel impulse responses
for link-level simulations.

### CoverageMap<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#coveragemap" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.rt.``CoverageMap`<a class="reference internal" href="../_modules/sionna/rt/coverage_map.html#CoverageMap">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap" title="Permalink to this definition"></a>
    
Stores the simulated coverage maps
    
A coverage map is generated for the loaded scene for every transmitter using
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.coverage_map" title="sionna.rt.Scene.coverage_map">`coverage_map()`</a>. Please refer to the documentation of this function
for further details.
    
An instance of this class can be indexed like a tensor of rank three with
shape `[num_tx,` `num_cells_y,` `num_cells_x]`, i.e.:
```python
cm = scene.coverage_map()
print(cm[0])      # prints the coverage map for transmitter 0
print(cm[0,1,2])  # prints the value of the cell (1,2) for transmitter 0
```

    
where `scene` is the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene" title="sionna.rt.Scene">`Scene`</a> loaded using
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.load_scene" title="sionna.rt.load_scene">`load_scene()`</a>.
<p class="rubric">Example
```python
import sionna
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver
scene = load_scene(sionna.rt.scene.munich)
# Configure antenna array for all transmitters
scene.tx_array = PlanarArray(num_rows=8,
                          num_cols=2,
                          vertical_spacing=0.7,
                          horizontal_spacing=0.5,
                          pattern="tr38901",
                          polarization="VH")
# Configure antenna array for all receivers
scene.rx_array = PlanarArray(num_rows=1,
                          num_cols=1,
                          vertical_spacing=0.5,
                          horizontal_spacing=0.5,
                          pattern="dipole",
                          polarization="cross")
# Add a transmitters
tx = Transmitter(name="tx",
              position=[8.5,21,30],
              orientation=[0,0,0])
scene.add(tx)
tx.look_at([40,80,1.5])
# Compute coverage map
cm = scene.coverage_map(max_depth=8)
# Show coverage map
cm.show()
```

<img alt="../_images/coverage_map_show.png" src="https://nvlabs.github.io/sionna/_images/coverage_map_show.png" />

`as_tensor`()<a class="reference internal" href="../_modules/sionna/rt/coverage_map.html#CoverageMap.as_tensor">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap.as_tensor" title="Permalink to this definition"></a>
    
Returns the coverage map as a tensor
Output
    
<em>[num_tx, num_cells_y, num_cells_x], tf.float</em> – The coverage map as a tensor




<em class="property">`property` </em>`cell_centers`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap.cell_centers" title="Permalink to this definition"></a>
    
Get the positions of the
centers of the cells in the global coordinate system
Type
    
[num_cells_y, num_cells_x, 3], tf.float




<em class="property">`property` </em>`cell_size`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap.cell_size" title="Permalink to this definition"></a>
    
Get the resolution of the coverage map, i.e., width
(in the local X direction) and height (in the local Y direction) in
of the cells of the coverage map
Type
    
[2], tf.float




<em class="property">`property` </em>`center`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap.center" title="Permalink to this definition"></a>
    
Get the center of the coverage map
Type
    
[3], tf.float




<em class="property">`property` </em>`num_cells_x`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap.num_cells_x" title="Permalink to this definition"></a>
    
Get the number of cells along the local X-axis
Type
    
int




<em class="property">`property` </em>`num_cells_y`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap.num_cells_y" title="Permalink to this definition"></a>
    
Get the number of cells along the local Y-axis
Type
    
int




<em class="property">`property` </em>`num_tx`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap.num_tx" title="Permalink to this definition"></a>
    
Get the number of transmitters
Type
    
int




<em class="property">`property` </em>`orientation`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap.orientation" title="Permalink to this definition"></a>
    
Get the orientation of the coverage map
Type
    
[3], tf.float




`sample_positions`(<em class="sig-param">`batch_size`</em>, <em class="sig-param">`tx``=``0`</em>, <em class="sig-param">`min_gain_db``=``None`</em>, <em class="sig-param">`max_gain_db``=``None`</em>, <em class="sig-param">`min_dist``=``None`</em>, <em class="sig-param">`max_dist``=``None`</em>, <em class="sig-param">`center_pos``=``False`</em>)<a class="reference internal" href="../_modules/sionna/rt/coverage_map.html#CoverageMap.sample_positions">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap.sample_positions" title="Permalink to this definition"></a>
    
Sample random user positions from a coverage map
    
For a given coverage map, `batch_size` random positions are sampled
such that the <em>expected</em>  path gain of this position is larger
than a given threshold `min_gain_db` or smaller than `max_gain_db`,
respectively.
Similarly, `min_dist` and `max_dist` define the minimum and maximum
distance of the random positions to the transmitter `tx`.
    
Note that due to the quantization of the coverage map into cells it is
not guaranteed that all above parameters are exactly fulfilled for a
returned position. This stems from the fact that every
individual cell of the coverage map describes the expected <em>average</em>
behavior of the surface within this cell. For instance, it may happen
that half of the selected cell is shadowed and, thus, no path to the
transmitter exists but the average path gain is still larger than the
given threshold. Please use `center_pos` = <cite>True</cite> to sample only
positions from the cell centers.
<img alt="../_images/cm_user_sampling.png" src="https://nvlabs.github.io/sionna/_images/cm_user_sampling.png" />
    
The above figure shows an example for random positions between 220m and
250m from the transmitter and a `max_gain_db` of -100 dB.
Keep in mind that the transmitter can have a different height than the
coverage map which also contributes to this distance.
For example if the transmitter is located 20m above the surface of the
coverage map and a `min_dist` of 20m is selected, also positions
directly below the transmitter are sampled.
Input
 
- **batch_size** (<em>int</em>) – Number of returned random positions
- **min_gain_db** (<em>float | None</em>) – Minimum path gain [dB]. Positions are only sampled from cells where
the path gain is larger or equal to this value.
Ignored if <cite>None</cite>.
Defaults to <cite>None</cite>.
- **max_gain_db** (<em>float | None</em>) – Maximum path gain [dB]. Positions are only sampled from cells where
the path gain is smaller or equal to this value.
Ignored if <cite>None</cite>.
Defaults to <cite>None</cite>.
- **min_dist** (<em>float | None</em>) – Minimum distance [m] from transmitter for all random positions.
Ignored if <cite>None</cite>.
Defaults to <cite>None</cite>.
- **max_dist** (<em>float | None</em>) – Maximum distance [m] from transmitter for all random positions.
Ignored if <cite>None</cite>.
Defaults to <cite>None</cite>.
- **tx** (<em>int | str</em>) – Index or name of the transmitter from whose coverage map
positions are sampled
- **center_pos** (<em>bool</em>) – If <cite>True</cite>, all returned positions are sampled from the cell center
(i.e., the grid of the coverage map). Otherwise, the positions are
randomly drawn from the surface of the cell.
Defaults to <cite>False</cite>.


Output
    
<em>[batch_size, 3], tf.float</em> – Random positions $(x,y,z)$ [m] that are in cells fulfilling the
above constraints w.r.t. distance and path gain




`show`(<em class="sig-param">`tx``=``0`</em>, <em class="sig-param">`vmin``=``None`</em>, <em class="sig-param">`vmax``=``None`</em>, <em class="sig-param">`show_tx``=``True`</em>)<a class="reference internal" href="../_modules/sionna/rt/coverage_map.html#CoverageMap.show">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap.show" title="Permalink to this definition"></a>
    
Visualizes a coverage map
    
The position of the transmitter is indicated by a red “+” marker.
Input
 
- **tx** (<em>int | str</em>) – Index or name of the transmitter for which to show the coverage map
Defaults to 0.
- **vmin,vmax** (float | <cite>None</cite>) – Define the range of path gains that the colormap covers.
If set to <cite>None</cite>, then covers the complete range.
Defaults to <cite>None</cite>.
- **show_tx** (<em>bool</em>) – If set to <cite>True</cite>, then the position of the transmitter is shown.
Defaults to <cite>True</cite>.


Output
    
`Figure` – Figure showing the coverage map




<em class="property">`property` </em>`size`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.CoverageMap.size" title="Permalink to this definition"></a>
    
Get the size of the coverage map
Type
    
[2], tf.float




## Cameras<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#cameras" title="Permalink to this headline"></a>
    
A <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a> defines a position and view direction
for rendering the scene.
<img alt="../_images/camera.png" src="https://nvlabs.github.io/sionna/_images/camera.png" />
    
The <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.cameras" title="sionna.rt.Scene.cameras">`cameras`</a> property of the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene" title="sionna.rt.Scene">`Scene`</a>
list all the cameras currently available for rendering. Cameras can be either
defined through the scene file or instantiated using the API.
The following code snippet shows how to load a scene and list the available
cameras:
```python
scene = load_scene(sionna.rt.scene.munich)
print(scene.cameras)
scene.render("scene-cam-0") # Use the first camera of the scene for rendering
```

<img alt="../_images/munich.png" src="https://nvlabs.github.io/sionna/_images/munich.png" />
    
A new camera can be instantiated as follows:
```python
cam = Camera("mycam", position=[200., 0.0, 50.])
scene.add(cam)
cam.look_at([0.0,0.0,0.0])
scene.render(cam) # Render using the Camera instance
scene.render("mycam") # or using the name of the camera
```
### Camera<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#camera" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.rt.``Camera`(<em class="sig-param">`name`</em>, <em class="sig-param">`position`</em>, <em class="sig-param">`orientation``=``[0.,` `0.,` `0.]`</em>, <em class="sig-param">`look_at``=``None`</em>)<a class="reference internal" href="../_modules/sionna/rt/camera.html#Camera">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="Permalink to this definition"></a>
    
A camera defines a position and view direction for rendering the scene.
    
In its local coordinate system, a camera looks toward the positive X-axis
with the positive Z-axis being the upward direction.
Input
 
- **name** (<em>str</em>) – Name.
Cannot be <cite>“preview”</cite>, as it is reserved for the viewpoint of the
interactive viewer.
- **position** (<em>[3], float</em>) – Position $(x,y,z)$ [m] as three-dimensional vector
- **orientation** (<em>[3], float</em>) – Orientation $(\alpha, \beta, \gamma)$ specified
through three angles corresponding to a 3D rotation
as defined in <a class="reference internal" href="../em_primer.html#equation-rotation">(3)</a>.
This parameter is ignored if `look_at` is not <cite>None</cite>.
Defaults to <cite>[0,0,0]</cite>.
- **look_at** ([3], float | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a> | None) – A position or instance of <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a>,
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a>, or <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a> to look at.
If set to <cite>None</cite>, then `orientation` is used to orientate the camera.




`look_at`(<em class="sig-param">`target`</em>)<a class="reference internal" href="../_modules/sionna/rt/camera.html#Camera.look_at">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera.look_at" title="Permalink to this definition"></a>
    
Sets the orientation so that the camera looks at a position, radio
device, or another camera.
    
Given a point $\mathbf{x}\in\mathbb{R}^3$ with spherical angles
$\theta$ and $\varphi$, the orientation of the camera
will be set equal to $(\varphi, \frac{\pi}{2}-\theta, 0.0)$.
Input
    
**target** ([3], float | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a> | str) – A position or the name or instance of a
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a>, or
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a> in the scene to look at.




<em class="property">`property` </em>`orientation`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera.orientation" title="Permalink to this definition"></a>
    
Get/set the orientation $(\alpha, \beta, \gamma)$
specified through three angles corresponding to a 3D rotation
as defined in <a class="reference internal" href="../em_primer.html#equation-rotation">(3)</a>.
Type
    
[3], float




<em class="property">`property` </em>`position`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera.position" title="Permalink to this definition"></a>
    
Get/set the position $(x,y,z)$ as three-dimensional
vector
Type
    
[3], float




## Scene Objects<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#scene-objects" title="Permalink to this headline"></a>
    
A scene is made of scene objects. Examples include cars, trees,
buildings, furniture, etc.
A scene object is characterized by its geometry and material (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial" title="sionna.rt.RadioMaterial">`RadioMaterial`</a>)
and implemented as an instance of the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.SceneObject" title="sionna.rt.SceneObject">`SceneObject`</a> class.
    
Scene objects are uniquely identified by their name.
To access a scene object, the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.get" title="sionna.rt.Scene.get">`get()`</a> method of
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene" title="sionna.rt.Scene">`Scene`</a> may be used.
For example, the following code snippet shows how to load a scene and list its scene objects:
```python
scene = load_scene(sionna.rt.scene.munich)
print(scene.objects)
```

    
To select an object, e.g., named <cite>“Schrannenhalle-itu_metal”</cite>, you can run:
```python
my_object = scene.get("Schrannenhalle-itu_metal")
```

    
You can then set the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial" title="sionna.rt.RadioMaterial">`RadioMaterial`</a>
of `my_object` as follows:
```python
my_object.radio_material = "itu_wood"
```

    
Most scene objects names have postfixes of the form “-material_name”. These are used during loading of a scene
to assign a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial" title="sionna.rt.RadioMaterial">`RadioMaterial`</a> to each of them. This <a class="reference external" href="https://youtu.be/7xHLDxUaQ7c">tutorial video</a>
explains how you can assign radio materials to objects when you create your own scenes.

### SceneObject<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sceneobject" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.rt.``SceneObject`<a class="reference internal" href="../_modules/sionna/rt/scene_object.html#SceneObject">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.SceneObject" title="Permalink to this definition"></a>
    
Every object in the scene is implemented by an instance of this class

<em class="property">`property` </em>`name`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.SceneObject.name" title="Permalink to this definition"></a>
    
Name
Type
    
str (read-only)




<em class="property">`property` </em>`radio_material`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.SceneObject.radio_material" title="Permalink to this definition"></a>
    
Get/set the radio material of the
object. Setting can be done by using either an instance of
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial" title="sionna.rt.RadioMaterial">`RadioMaterial`</a> or the material name (<cite>str</cite>).
If the radio material is not part of the scene, it will be added. This
can raise an error if a different radio material with the same name was
already added to the scene.
Type
    
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial" title="sionna.rt.RadioMaterial">`RadioMaterial`</a>




## Radio Materials<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#radio-materials" title="Permalink to this headline"></a>
    
A <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial" title="sionna.rt.RadioMaterial">`RadioMaterial`</a> contains everything that is needed to enable the simulation
of the interaction of a radio wave with an object made of a particular material.
More precisely, it consists of the real-valued relative permittivity $\varepsilon_r$,
the conductivity $\sigma$, and the relative
permeability $\mu_r$. For more details, see <a class="reference internal" href="../em_primer.html#equation-epsilon">(7)</a>, <a class="reference internal" href="../em_primer.html#equation-mu">(8)</a>, <a class="reference internal" href="../em_primer.html#equation-eta">(9)</a>.
These quantities can possibly depend on the frequency of the incident radio
wave. Note that Sionna currently only allows non-magnetic materials with $\mu_r=1$.
    
Additionally, a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial" title="sionna.rt.RadioMaterial">`RadioMaterial`</a> can have an effective roughness (ER)
associated with it, leading to diffuse reflections (see, e.g., <a class="reference internal" href="../em_primer.html#degli-esposti11" id="id15">[Degli-Esposti11]</a>).
The ER model requires a scattering coefficient $S\in[0,1]$ <a class="reference internal" href="../em_primer.html#equation-scattering-coefficient">(37)</a>,
a cross-polarization discrimination coefficient $K_x$ <a class="reference internal" href="../em_primer.html#equation-xpd">(39)</a>, as well as a scattering pattern
$f_\text{s}(\hat{\mathbf{k}}_\text{i}, \hat{\mathbf{k}}_\text{s})$ <a class="reference internal" href="../em_primer.html#equation-lambertian-model">(40)</a>–<a class="reference internal" href="../em_primer.html#equation-backscattering-model">(42)</a>, such as the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.LambertianPattern" title="sionna.rt.LambertianPattern">`LambertianPattern`</a> or <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.DirectivePattern" title="sionna.rt.DirectivePattern">`DirectivePattern`</a>. The meaning of
these parameters is explained in <a class="reference internal" href="../em_primer.html#scattering">Scattering</a>.
    
Similarly to scene objects (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.SceneObject" title="sionna.rt.SceneObject">`SceneObject`</a>), all radio
materials are uniquely identified by their name.
For example, specifying that a scene object named <cite>“wall”</cite> is made of the
material named <cite>“itu-brick”</cite> is done as follows:
```python
obj = scene.get("wall") # obj is a SceneObject
obj.radio_material = "itu_brick" # "wall" is made of "itu_brick"
```

    
Sionna provides the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#provided-materials">ITU models of several materials</a> whose properties
are automatically updated according to the configured <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.frequency" title="sionna.rt.Scene.frequency">`frequency`</a>.
It is also possible to
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#custom-radio-materials">define custom radio materials</a>.
<p id="provided-materials">**Radio materials provided with Sionna**
    
Sionna provides the models of all of the materials defined in the ITU-R P.2040-2
recommendation <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#itur-p2040-2" id="id16">[ITUR_P2040_2]</a>. These models are based on curve fitting to
measurement results and assume non-ionized and non-magnetic materials
($\mu_r = 1$).
Frequency dependence is modeled by

$$
\begin{split}\begin{align}
   \varepsilon_r &= a f_{\text{GHz}}^b\\
   \sigma &= c f_{\text{GHz}}^d
\end{align}\end{split}
$$
    
where $f_{\text{GHz}}$ is the frequency in GHz, and the constants
$a$, $b$, $c$, and $d$ characterize the material.
The table below provides their values which are used in Sionna
(from <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#itur-p2040-2" id="id17">[ITUR_P2040_2]</a>).
Note that the relative permittivity $\varepsilon_r$ and
conductivity $\sigma$ of all materials are updated automatically when
the frequency is set through the scene’s property <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.frequency" title="sionna.rt.Scene.frequency">`frequency`</a>.
Moreover, by default, the scattering coefficient, $S$, of these materials is set to
0, leading to no diffuse reflection.
<table class="docutils align-default">
<colgroup>
<col style="width: 25%" />
<col style="width: 17%" />
<col style="width: 15%" />
<col style="width: 14%" />
<col style="width: 9%" />
<col style="width: 21%" />
</colgroup>
<tbody>
<tr class="row-odd"><td rowspan="2">    
Material name</td>
<td colspan="2">    
Real part of relative permittivity</td>
<td colspan="2">    
Conductivity [S/m]</td>
<td rowspan="2">    
Frequency range (GHz)</td>
</tr>
<tr class="row-even"><td>    
a</td>
<td>    
b</td>
<td>    
c</td>
<td>    
d</td>
</tr>
<tr class="row-odd"><td>    
vacuum</td>
<td>    
1</td>
<td>    
0</td>
<td>    
0</td>
<td>    
0</td>
<td>    
0.001 – 100</td>
</tr>
<tr class="row-even"><td>    
itu_concrete</td>
<td>    
5.24</td>
<td>    
0</td>
<td>    
0.0462</td>
<td>    
0.7822</td>
<td>    
1 – 100</td>
</tr>
<tr class="row-odd"><td>    
itu_brick</td>
<td>    
3.91</td>
<td>    
0</td>
<td>    
0.0238</td>
<td>    
0.16</td>
<td>    
1 – 40</td>
</tr>
<tr class="row-even"><td>    
itu_plasterboard</td>
<td>    
2.73</td>
<td>    
0</td>
<td>    
0.0085</td>
<td>    
0.9395</td>
<td>    
1 – 100</td>
</tr>
<tr class="row-odd"><td>    
itu_wood</td>
<td>    
1.99</td>
<td>    
0</td>
<td>    
0.0047</td>
<td>    
1.0718</td>
<td>    
0.001 – 100</td>
</tr>
<tr class="row-even"><td rowspan="2">    
itu_glass</td>
<td>    
6.31</td>
<td>    
0</td>
<td>    
0.0036</td>
<td>    
1.3394</td>
<td>    
0.1 – 100</td>
</tr>
<tr class="row-odd"><td>    
5.79</td>
<td>    
0</td>
<td>    
0.0004</td>
<td>    
1.658</td>
<td>    
220 – 450</td>
</tr>
<tr class="row-even"><td rowspan="2">    
itu_ceiling_board</td>
<td>    
1.48</td>
<td>    
0</td>
<td>    
0.0011</td>
<td>    
1.0750</td>
<td>    
1 – 100</td>
</tr>
<tr class="row-odd"><td>    
1.52</td>
<td>    
0</td>
<td>    
0.0029</td>
<td>    
1.029</td>
<td>    
220 – 450</td>
</tr>
<tr class="row-even"><td>    
itu_chipboard</td>
<td>    
2.58</td>
<td>    
0</td>
<td>    
0.0217</td>
<td>    
0.7800</td>
<td>    
1 – 100</td>
</tr>
<tr class="row-odd"><td>    
itu_plywood</td>
<td>    
2.71</td>
<td>    
0</td>
<td>    
0.33</td>
<td>    
0</td>
<td>    
1 – 40</td>
</tr>
<tr class="row-even"><td>    
itu_marble</td>
<td>    
7.074</td>
<td>    
0</td>
<td>    
0.0055</td>
<td>    
0.9262</td>
<td>    
1 – 60</td>
</tr>
<tr class="row-odd"><td>    
itu_floorboard</td>
<td>    
3.66</td>
<td>    
0</td>
<td>    
0.0044</td>
<td>    
1.3515</td>
<td>    
50 – 100</td>
</tr>
<tr class="row-even"><td>    
itu_metal</td>
<td>    
1</td>
<td>    
0</td>
<td>    
$10^7$</td>
<td>    
0</td>
<td>    
1 – 100</td>
</tr>
<tr class="row-odd"><td>    
itu_very_dry_ground</td>
<td>    
3</td>
<td>    
0</td>
<td>    
0.00015</td>
<td>    
2.52</td>
<td>    
1 – 10</td>
</tr>
<tr class="row-even"><td>    
itu_medium_dry_ground</td>
<td>    
15</td>
<td>    
-0.1</td>
<td>    
0.035</td>
<td>    
1.63</td>
<td>    
1 – 10</td>
</tr>
<tr class="row-odd"><td>    
itu_wet_ground</td>
<td>    
30</td>
<td>    
-0.4</td>
<td>    
0.15</td>
<td>    
1.30</td>
<td>    
1 – 10</td>
</tr>
</tbody>
</table>
<p id="custom-radio-materials">**Defining custom radio materials**
    
Custom radio materials can be implemented using the
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial" title="sionna.rt.RadioMaterial">`RadioMaterial`</a> class by specifying a relative permittivity
$\varepsilon_r$ and conductivity $\sigma$, as well as optional
parameters related to diffuse scattering, such as the scattering coefficient $S$,
cross-polarization discrimination coefficient $K_x$, and scattering pattern $f_\text{s}(\hat{\mathbf{k}}_\text{i}, \hat{\mathbf{k}}_\text{s})$.
Note that only non-magnetic materials with $\mu_r=1$ are currently allowed.
The following code snippet shows how to create a custom radio material.
```python
load_scene() # Load empty scene
custom_material = RadioMaterial("my_material",
                                relative_permittivity=2.0,
                                conductivity=5.0,
                                scattering_coefficient=0.3,
                                xpd_coefficient=0.1,
                                scattering_pattern=LambertianPattern())
```

    
It is also possible to define the properties of a material through a callback
function that computes the material properties
$(\varepsilon_r, \sigma)$ from the frequency:
```python
def my_material_callback(f_hz):
   relative_permittivity = compute_relative_permittivity(f_hz)
   conductivity = compute_conductivity(f_hz)
   return (relative_permittivity, conductivity)
custom_material = RadioMaterial("my_material",
                                frequency_update_callback=my_material_callback)
scene.add(custom_material)
```

    
Once defined, the custom material can be assigned to a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.SceneObject" title="sionna.rt.SceneObject">`SceneObject`</a> using its name:
```python
obj = scene.get("my_object") # obj is a SceneObject
obj.radio_material = "my_material" # "my_object" is made of "my_material"
```

    
or the material instance:
```python
obj = scene.get("my_object") # obj is a SceneObject
obj.radio_material = custom_material # "my_object" is made of "my_material"
```

    
The material parameters can be assigned to TensorFlow variables or tensors, such as
the output of a Keras layer defining a neural network. This allows one to make materials
trainable:
```python
mat = RadioMaterial("my_mat",
                    relative_permittivity= tf.Variable(2.1, dtype=tf.float32))
mat.conductivity = tf.Variable(0.0, dtype=tf.float32)
```
### RadioMaterial<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#radiomaterial" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.rt.``RadioMaterial`(<em class="sig-param">`name`</em>, <em class="sig-param">`relative_permittivity``=``1.0`</em>, <em class="sig-param">`conductivity``=``0.0`</em>, <em class="sig-param">`scattering_coefficient``=``0.0`</em>, <em class="sig-param">`xpd_coefficient``=``0.0`</em>, <em class="sig-param">`scattering_pattern``=``None`</em>, <em class="sig-param">`frequency_update_callback``=``None`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/rt/radio_material.html#RadioMaterial">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial" title="Permalink to this definition"></a>
    
Class implementing a radio material
    
A radio material is defined by its relative permittivity
$\varepsilon_r$ and conductivity $\sigma$ (see <a class="reference internal" href="../em_primer.html#equation-eta">(9)</a>),
as well as optional parameters related to diffuse scattering, such as the
scattering coefficient $S$, cross-polarization discrimination
coefficient $K_x$, and scattering pattern $f_\text{s}(\hat{\mathbf{k}}_\text{i}, \hat{\mathbf{k}}_\text{s})$.
    
We assume non-ionized and non-magnetic materials, and therefore the
permeability $\mu$ of the material is assumed to be equal
to the permeability of vacuum i.e., $\mu_r=1.0$.
    
For frequency-dependent materials, it is possible to
specify a callback function `frequency_update_callback` that computes
the material properties $(\varepsilon_r, \sigma)$ from the
frequency. If a callback function is specified, the material properties
cannot be set and the values specified at instantiation are ignored.
The callback should return <cite>-1</cite> for both the relative permittivity and
the conductivity if these are not defined for the given carrier frequency.
    
The material properties can be assigned to a TensorFlow variable or
tensor. In the latter case, the tensor could be the output of a callable,
such as a Keras layer implementing a neural network. In the former case, it
could be set to a trainable variable:
```python
mat = RadioMaterial("my_mat")
mat.conductivity = tf.Variable(0.0, dtype=tf.float32)
```

Parameters
 
- **name** (<em>str</em>) – Unique name of the material
- **relative_permittivity** (float | <cite>None</cite>) – Relative permittivity of the material.
Must be larger or equal to 1.
Defaults to 1. Ignored if `frequency_update_callback`
is provided.
- **conductivity** (float | <cite>None</cite>) – Conductivity of the material [S/m].
Must be non-negative.
Defaults to 0.
Ignored if `frequency_update_callback`
is provided.
- **scattering_coefficient** (<em>float</em>) – Scattering coefficient $S\in[0,1]$ as defined in
<a class="reference internal" href="../em_primer.html#equation-scattering-coefficient">(37)</a>.
Defaults to 0.
- **xpd_coefficient** (<em>float</em>) – Cross-polarization discrimination coefficient $K_x\in[0,1]$ as
defined in <a class="reference internal" href="../em_primer.html#equation-xpd">(39)</a>.
Only relevant if `scattering_coefficient`>0.
Defaults to 0.
- **scattering_pattern** (<em>ScatteringPattern</em>) – `ScatteringPattern` to be applied.
Only relevant if `scattering_coefficient`>0.
Defaults to <cite>None</cite>, which implies a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.LambertianPattern" title="sionna.rt.LambertianPattern">`LambertianPattern`</a>.
- **frequency_update_callback** (callable | <cite>None</cite>) –     
An optional callable object used to obtain the material parameters
from the scene’s <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.frequency" title="sionna.rt.Scene.frequency">`frequency`</a>.
This callable must take as input the frequency [Hz] and
must return the material properties as a tuple:
    
`(relative_permittivity,` `conductivity)`.
    
If set to <cite>None</cite>, the material properties are constant and equal
to `relative_permittivity` and `conductivity`.
Defaults to <cite>None</cite>.

- **dtype** (<em>tf.complex64</em><em> or </em><em>tf.complex128</em>) – Datatype.
Defaults to <cite>tf.complex64</cite>.




<em class="property">`property` </em>`complex_relative_permittivity`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial.complex_relative_permittivity" title="Permalink to this definition"></a>
    
Complex relative permittivity
$\eta$ <a class="reference internal" href="../em_primer.html#equation-eta">(9)</a>
Type
    
tf.complex (read-only)




<em class="property">`property` </em>`conductivity`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial.conductivity" title="Permalink to this definition"></a>
    
Get/set the conductivity
$\sigma$ [S/m] <a class="reference internal" href="../em_primer.html#equation-eta">(9)</a>
Type
    
tf.float




<em class="property">`property` </em>`frequency_update_callback`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial.frequency_update_callback" title="Permalink to this definition"></a>
    
Get/set frequency update callback function
Type
    
callable




<em class="property">`property` </em>`is_used`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial.is_used" title="Permalink to this definition"></a>
    
Indicator if the material is used by at least one object of
the scene
Type
    
bool




<em class="property">`property` </em>`name`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial.name" title="Permalink to this definition"></a>
    
Name of the radio material
Type
    
str (read-only)




<em class="property">`property` </em>`relative_permeability`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial.relative_permeability" title="Permalink to this definition"></a>
    
Relative permeability
$\mu_r$ <a class="reference internal" href="../em_primer.html#equation-mu">(8)</a>.
Defaults to 1.
Type
    
tf.float (read-only)




<em class="property">`property` </em>`relative_permittivity`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial.relative_permittivity" title="Permalink to this definition"></a>
    
Get/set the relative permittivity
$\varepsilon_r$ <a class="reference internal" href="../em_primer.html#equation-eta">(9)</a>
Type
    
tf.float




<em class="property">`property` </em>`scattering_coefficient`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial.scattering_coefficient" title="Permalink to this definition"></a>
    
Get/set the scattering coefficient
$S\in[0,1]$ <a class="reference internal" href="../em_primer.html#equation-scattering-coefficient">(37)</a>.
Type
    
tf.float




<em class="property">`property` </em>`scattering_pattern`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial.scattering_pattern" title="Permalink to this definition"></a>
    
Get/set the ScatteringPattern.
Type
    
ScatteringPattern




<em class="property">`property` </em>`use_counter`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial.use_counter" title="Permalink to this definition"></a>
    
Number of scene objects using this material
Type
    
int




<em class="property">`property` </em>`using_objects`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial.using_objects" title="Permalink to this definition"></a>
    
Identifiers of the objects using this
material
Type
    
[num_using_objects], tf.int




<em class="property">`property` </em>`well_defined`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial.well_defined" title="Permalink to this definition"></a>
    
Get if the material is well-defined
Type
    
bool




<em class="property">`property` </em>`xpd_coefficient`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.RadioMaterial.xpd_coefficient" title="Permalink to this definition"></a>
    
Get/set the cross-polarization discrimination coefficient
$K_x\in[0,1]$ <a class="reference internal" href="../em_primer.html#equation-xpd">(39)</a>.
Type
    
tf.float




### ScatteringPattern<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#scatteringpattern" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.rt.``LambertianPattern`(<em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/rt/scattering_pattern.html#LambertianPattern">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.LambertianPattern" title="Permalink to this definition"></a>
    
Lambertian scattering model from <a class="reference internal" href="../em_primer.html#degli-esposti07" id="id18">[Degli-Esposti07]</a> as given in <a class="reference internal" href="../em_primer.html#equation-lambertian-model">(40)</a>
Parameters
    
**dtype** (<em>tf.complex64</em><em> or </em><em>tf.complex128</em>) – Datatype used for all computations.
Defaults to <cite>tf.complex64</cite>.

Input
 
- **k_i** (<em>[batch_size, 3], dtype.real_dtype</em>) – Incoming directions
- **k_s** (<em>[batch_size,3], dtype.real_dtype</em>) – Outgoing directions


Output
    
**pattern** (<em>[batch_size], dtype.real_dtype</em>) – Scattering pattern


<p class="rubric">Example
```python
>>> LambertianPattern().visualize()
```

<img alt="../_images/lambertian_pattern_3d.png" src="https://nvlabs.github.io/sionna/_images/lambertian_pattern_3d.png" />

<img alt="../_images/lambertian_pattern_cut.png" src="https://nvlabs.github.io/sionna/_images/lambertian_pattern_cut.png" />

`visualize`(<em class="sig-param">`k_i``=``(0.7071,` `0.0,` `-` `0.7071)`</em>, <em class="sig-param">`show_directions``=``False`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.LambertianPattern.visualize" title="Permalink to this definition"></a>
    
Visualizes the scattering pattern
    
It is assumed that the surface normal points toward the
positive z-axis.
Input
 
- **k_i** (<em>[3], array_like</em>) – Incoming direction
- **show_directions** (<em>bool</em>) – If <cite>True</cite>, the incoming and specular reflection directions
are shown.
Defaults to <cite>False</cite>.


Output
 
- `matplotlib.pyplot.Figure` – 3D visualization of the scattering pattern
- `matplotlib.pyplot.Figure` – Visualization of the incident plane cut through
the scattering pattern






<em class="property">`class` </em>`sionna.rt.``DirectivePattern`(<em class="sig-param">`alpha_r`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/rt/scattering_pattern.html#DirectivePattern">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.DirectivePattern" title="Permalink to this definition"></a>
    
Directive scattering model from <a class="reference internal" href="../em_primer.html#degli-esposti07" id="id19">[Degli-Esposti07]</a> as given in <a class="reference internal" href="../em_primer.html#equation-directive-model">(41)</a>
Parameters
 
- **alpha_r** (<em>int</em><em>, </em><em>[</em><em>1</em><em>,</em><em>2</em><em>,</em><em>...</em><em>]</em>) – Parameter related to the width of the scattering lobe in the
direction of the specular reflection.
- **dtype** (<em>tf.complex64</em><em> or </em><em>tf.complex128</em>) – Datatype used for all computations.
Defaults to <cite>tf.complex64</cite>.


Input
 
- **k_i** (<em>[batch_size, 3], dtype.real_dtype</em>) – Incoming directions
- **k_s** (<em>[batch_size,3], dtype.real_dtype</em>) – Outgoing directions


Output
    
**pattern** (<em>[batch_size], dtype.real_dtype</em>) – Scattering pattern


<p class="rubric">Example
```python
>>> DirectivePattern(alpha_r=10).visualize()
```

<img alt="../_images/directive_pattern_3d.png" src="https://nvlabs.github.io/sionna/_images/directive_pattern_3d.png" />

<img alt="../_images/directive_pattern_cut.png" src="https://nvlabs.github.io/sionna/_images/directive_pattern_cut.png" />

<em class="property">`property` </em>`alpha_r`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.DirectivePattern.alpha_r" title="Permalink to this definition"></a>
    
Get/set `alpha_r`
Type
    
bool




`visualize`(<em class="sig-param">`k_i``=``(0.7071,` `0.0,` `-` `0.7071)`</em>, <em class="sig-param">`show_directions``=``False`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.DirectivePattern.visualize" title="Permalink to this definition"></a>
    
Visualizes the scattering pattern
    
It is assumed that the surface normal points toward the
positive z-axis.
Input
 
- **k_i** (<em>[3], array_like</em>) – Incoming direction
- **show_directions** (<em>bool</em>) – If <cite>True</cite>, the incoming and specular reflection directions
are shown.
Defaults to <cite>False</cite>.


Output
 
- `matplotlib.pyplot.Figure` – 3D visualization of the scattering pattern
- `matplotlib.pyplot.Figure` – Visualization of the incident plane cut through
the scattering pattern






<em class="property">`class` </em>`sionna.rt.``BackscatteringPattern`(<em class="sig-param">`alpha_r`</em>, <em class="sig-param">`alpha_i`</em>, <em class="sig-param">`lambda_`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/rt/scattering_pattern.html#BackscatteringPattern">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.BackscatteringPattern" title="Permalink to this definition"></a>
    
Backscattering model from <a class="reference internal" href="../em_primer.html#degli-esposti07" id="id20">[Degli-Esposti07]</a> as given in <a class="reference internal" href="../em_primer.html#equation-backscattering-model">(42)</a>
    
The parameter `lambda_` can be assigned to a TensorFlow variable
or tensor.  In the latter case, the tensor can be the output of a callable, such as
a Keras layer implementing a neural network.
In the former case, it can be set to a trainable variable:
```python
sp = BackscatteringPattern(alpha_r=3,
                           alpha_i=5,
                           lambda_=tf.Variable(0.3, dtype=tf.float32))
```

Parameters
 
- **alpha_r** (<em>int</em><em>, </em><em>[</em><em>1</em><em>,</em><em>2</em><em>,</em><em>...</em><em>]</em>) – Parameter related to the width of the scattering lobe in the
direction of the specular reflection.
- **alpha_i** (<em>int</em><em>, </em><em>[</em><em>1</em><em>,</em><em>2</em><em>,</em><em>...</em><em>]</em>) – Parameter related to the width of the scattering lobe in the
incoming direction.
- **lambda** (<em>float</em><em>, </em><em>[</em><em>0</em><em>,</em><em>1</em><em>]</em>) – Parameter determining the percentage of the diffusely
reflected energy in the lobe around the specular reflection.
- **dtype** (<em>tf.complex64</em><em> or </em><em>tf.complex128</em>) – Datatype used for all computations.
Defaults to <cite>tf.complex64</cite>.


Input
 
- **k_i** (<em>[batch_size, 3], dtype.real_dtype</em>) – Incoming directions
- **k_s** (<em>[batch_size,3], dtype.real_dtype</em>) – Outgoing directions


Output
    
**pattern** (<em>[batch_size], dtype.real_dtype</em>) – Scattering pattern


<p class="rubric">Example
```python
>>> BackscatteringPattern(alpha_r=20, alpha_i=30, lambda_=0.7).visualize()
```

<img alt="../_images/backscattering_pattern_3d.png" src="https://nvlabs.github.io/sionna/_images/backscattering_pattern_3d.png" />

<img alt="../_images/backscattering_pattern_cut.png" src="https://nvlabs.github.io/sionna/_images/backscattering_pattern_cut.png" />

<em class="property">`property` </em>`alpha_i`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.BackscatteringPattern.alpha_i" title="Permalink to this definition"></a>
    
Get/set `alpha_i`
Type
    
bool




<em class="property">`property` </em>`alpha_r`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.BackscatteringPattern.alpha_r" title="Permalink to this definition"></a>
    
Get/set `alpha_r`
Type
    
bool




<em class="property">`property` </em>`lambda_`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.BackscatteringPattern.lambda_" title="Permalink to this definition"></a>
    
Get/set `lambda_`
Type
    
bool




`visualize`(<em class="sig-param">`k_i``=``(0.7071,` `0.0,` `-` `0.7071)`</em>, <em class="sig-param">`show_directions``=``False`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.BackscatteringPattern.visualize" title="Permalink to this definition"></a>
    
Visualizes the scattering pattern
    
It is assumed that the surface normal points toward the
positive z-axis.
Input
 
- **k_i** (<em>[3], array_like</em>) – Incoming direction
- **show_directions** (<em>bool</em>) – If <cite>True</cite>, the incoming and specular reflection directions
are shown.
Defaults to <cite>False</cite>.


Output
 
- `matplotlib.pyplot.Figure` – 3D visualization of the scattering pattern
- `matplotlib.pyplot.Figure` – Visualization of the incident plane cut through
the scattering pattern





## Radio Devices<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#radio-devices" title="Permalink to this headline"></a>
    
A radio device refers to a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a> or <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a> equipped
with an <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.AntennaArray" title="sionna.rt.AntennaArray">`AntennaArray`</a> as specified by the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene" title="sionna.rt.Scene">`Scene`</a>’s properties
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.tx_array" title="sionna.rt.Scene.tx_array">`tx_array`</a> and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.rx_array" title="sionna.rt.Scene.rx_array">`rx_array`</a>, respectively.
    
The following code snippet shows how to instantiate a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a>
equipped with a $4 \times 2$ <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.PlanarArray" title="sionna.rt.PlanarArray">`PlanarArray`</a> with cross-polarized isotropic antennas:
```python
 scene.tx_array = PlanarArray(num_rows=4,
                              num_cols=2,
                              vertical_spacing=0.5,
                              horizontal_spacing=0.5,
                              pattern="iso",
                              polarization="cross")
 my_tx = Transmitter(name="my_tx",
                     position=(0,0,0),
                     orientation=(0,0,0))
scene.add(my_tx)
```

    
The position $(x,y,z)$ and orientation $(\alpha, \beta, \gamma)$ of a radio device
can be freely configured. The latter is specified through three angles corresponding to a 3D
rotation as defined in <a class="reference internal" href="../em_primer.html#equation-rotation">(3)</a>.
Both can be assigned to TensorFlow variables or tensors. In the latter case,
the tensor can be the output of a callable, such as a Keras layer implementing a neural network.
In the former case, it can be set to a trainable variable.
    
Radio devices need to be explicitly added to the scene using the scene’s method <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.add" title="sionna.rt.Scene.add">`add()`</a>
and can be removed from it using <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.remove" title="sionna.rt.Scene.remove">`remove()`</a>:
```python
scene = load_scene()
scene.add(Transmitter("tx", [10.0, 0.0, 1.5], [0.0,0.0,0.0]))
scene.remove("tx")
```
### Transmitter<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#transmitter" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.rt.``Transmitter`(<em class="sig-param">`name`</em>, <em class="sig-param">`position`</em>, <em class="sig-param">`orientation``=``(0.0,` `0.0,` `0.0)`</em>, <em class="sig-param">`look_at``=``None`</em>, <em class="sig-param">`color``=``(0.16,` `0.502,` `0.725)`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/rt/transmitter.html#Transmitter">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="Permalink to this definition"></a>
    
Class defining a transmitter
    
The `position` and `orientation` properties can be assigned to a TensorFlow
variable or tensor. In the latter case, the tensor can be the output of a callable,
such as a Keras layer implementing a neural network. In the former case, it
can be set to a trainable variable:
```python
tx = Transmitter(name="my_tx",
                 position=tf.Variable([0, 0, 0], dtype=tf.float32),
                 orientation=tf.Variable([0, 0, 0], dtype=tf.float32))
```

Parameters
 
- **name** (<em>str</em>) – Name
- **position** (<em>[</em><em>3</em><em>]</em><em>, </em><em>float</em>) – Position $(x,y,z)$ [m] as three-dimensional vector
- **orientation** (<em>[</em><em>3</em><em>]</em><em>, </em><em>float</em>) – Orientation $(\alpha, \beta, \gamma)$ [rad] specified
through three angles corresponding to a 3D rotation
as defined in <a class="reference internal" href="../em_primer.html#equation-rotation">(3)</a>.
This parameter is ignored if `look_at` is not <cite>None</cite>.
Defaults to [0,0,0].
- **look_at** ([3], float | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a> | None) – A position or the instance of a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a>,
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a>, or <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a> to look at.
If set to <cite>None</cite>, then `orientation` is used to orientate the device.
- **color** (<em>[</em><em>3</em><em>]</em><em>, </em><em>float</em>) – Defines the RGB (red, green, blue) `color` parameter for the device as displayed in the previewer and renderer.
Each RGB component must have a value within the range $\in [0,1]$.
Defaults to <cite>[0.160, 0.502, 0.725]</cite>.
- **dtype** (<em>tf.complex</em>) – Datatype to be used in internal calculations.
Defaults to <cite>tf.complex64</cite>.




<em class="property">`property` </em>`color`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter.color" title="Permalink to this definition"></a>
    
Get/set the the RGB (red, green, blue) color for the device as displayed in the previewer and renderer.
Each RGB component must have a value within the range $\in [0,1]$.
Type
    
[3], float




`look_at`(<em class="sig-param">`target`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter.look_at" title="Permalink to this definition"></a>
    
Sets the orientation so that the x-axis points toward a
position, radio device, or camera.
    
Given a point $\mathbf{x}\in\mathbb{R}^3$ with spherical angles
$\theta$ and $\varphi$, the orientation of the radio device
will be set equal to $(\varphi, \frac{\pi}{2}-\theta, 0.0)$.
Input
    
**target** ([3], float | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a> | str) – A position or the name or instance of a
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a>, or
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a> in the scene to look at.




<em class="property">`property` </em>`name`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter.name" title="Permalink to this definition"></a>
    
Name
Type
    
str (read-only)




<em class="property">`property` </em>`orientation`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter.orientation" title="Permalink to this definition"></a>
    
Get/set the orientation
Type
    
[3], tf.float




<em class="property">`property` </em>`position`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter.position" title="Permalink to this definition"></a>
    
Get/set the position
Type
    
[3], tf.float




### Receiver<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#receiver" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.rt.``Receiver`(<em class="sig-param">`name`</em>, <em class="sig-param">`position`</em>, <em class="sig-param">`orientation``=``(0.0,` `0.0,` `0.0)`</em>, <em class="sig-param">`look_at``=``None`</em>, <em class="sig-param">`color``=``(0.153,` `0.682,` `0.375)`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/rt/receiver.html#Receiver">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="Permalink to this definition"></a>
    
Class defining a receiver
    
The `position` and `orientation` properties can be assigned to a TensorFlow
variable or tensor. In the latter case, the tensor can be the output of a callable,
such as a Keras layer implementing a neural network. In the former case, it
can be set to a trainable variable:
```python
rx = Transmitter(name="my_rx",
                 position=tf.Variable([0, 0, 0], dtype=tf.float32),
                 orientation=tf.Variable([0, 0, 0], dtype=tf.float32))
```

Parameters
 
- **name** (<em>str</em>) – Name
- **position** (<em>[</em><em>3</em><em>]</em><em>, </em><em>float</em>) – Position $(x,y,z)$ as three-dimensional vector
- **orientation** (<em>[</em><em>3</em><em>]</em><em>, </em><em>float</em>) – Orientation $(\alpha, \beta, \gamma)$ [rad] specified
through three angles corresponding to a 3D rotation
as defined in <a class="reference internal" href="../em_primer.html#equation-rotation">(3)</a>.
This parameter is ignored if `look_at` is not <cite>None</cite>.
Defaults to [0,0,0].
- **look_at** ([3], float | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a> | None) – A position or the instance of a <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a>,
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a>, or <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a> to look at.
If set to <cite>None</cite>, then `orientation` is used to orientate the device.
- **color** (<em>[</em><em>3</em><em>]</em><em>, </em><em>float</em>) – Defines the RGB (red, green, blue) `color` parameter for the device as displayed in the previewer and renderer.
Each RGB component must have a value within the range $\in [0,1]$.
Defaults to <cite>[0.153, 0.682, 0.375]</cite>.
- **dtype** (<em>tf.complex</em>) – Datatype to be used in internal calculations.
Defaults to <cite>tf.complex64</cite>.




<em class="property">`property` </em>`color`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver.color" title="Permalink to this definition"></a>
    
Get/set the the RGB (red, green, blue) color for the device as displayed in the previewer and renderer.
Each RGB component must have a value within the range $\in [0,1]$.
Type
    
[3], float




`look_at`(<em class="sig-param">`target`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver.look_at" title="Permalink to this definition"></a>
    
Sets the orientation so that the x-axis points toward a
position, radio device, or camera.
    
Given a point $\mathbf{x}\in\mathbb{R}^3$ with spherical angles
$\theta$ and $\varphi$, the orientation of the radio device
will be set equal to $(\varphi, \frac{\pi}{2}-\theta, 0.0)$.
Input
    
**target** ([3], float | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a> | <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a> | str) – A position or the name or instance of a
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a>, <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a>, or
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Camera" title="sionna.rt.Camera">`Camera`</a> in the scene to look at.




<em class="property">`property` </em>`name`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver.name" title="Permalink to this definition"></a>
    
Name
Type
    
str (read-only)




<em class="property">`property` </em>`orientation`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver.orientation" title="Permalink to this definition"></a>
    
Get/set the orientation
Type
    
[3], tf.float




<em class="property">`property` </em>`position`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver.position" title="Permalink to this definition"></a>
    
Get/set the position
Type
    
[3], tf.float




## Antenna Arrays<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#antenna-arrays" title="Permalink to this headline"></a>
    
Transmitters (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a>) and receivers (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a>) are equipped with an <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.AntennaArray" title="sionna.rt.AntennaArray">`AntennaArray`</a> that is composed of one or more antennas. All transmitters and all receivers share the same <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.AntennaArray" title="sionna.rt.AntennaArray">`AntennaArray`</a> which can be set through the scene properties <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.tx_array" title="sionna.rt.Scene.tx_array">`tx_array`</a> and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.rx_array" title="sionna.rt.Scene.rx_array">`rx_array`</a>, respectively.

### AntennaArray<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#antennaarray" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.rt.``AntennaArray`(<em class="sig-param">`antenna`</em>, <em class="sig-param">`positions`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/rt/antenna_array.html#AntennaArray">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.AntennaArray" title="Permalink to this definition"></a>
    
Class implementing an antenna array
    
An antenna array is composed of identical antennas that are placed
at different positions. The `positions` parameter can be assigned
to a TensorFlow variable or tensor.
```python
array = AntennaArray(antenna=Antenna("tr38901", "V"),
                     positions=tf.Variable([[0,0,0], [0, 1, 1]]))
```

Parameters
 
- **antenna** (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Antenna" title="sionna.rt.Antenna">`Antenna`</a>) – Antenna instance
- **positions** (<em>[</em><em>array_size</em><em>, </em><em>3</em><em>]</em><em>, </em><em>array_like</em>) – Array of relative positions $(x,y,z)$ [m] of each
antenna (dual-polarized antennas are counted as a single antenna
and share the same position).
The absolute position of the antennas is obtained by
adding the position of the <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a>
or <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a> using it.
- **dtype** (<em>tf.complex64</em><em> or </em><em>tf.complex128</em>) – Data type used for all computations.
Defaults to <cite>tf.complex64</cite>.




<em class="property">`property` </em>`antenna`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.AntennaArray.antenna" title="Permalink to this definition"></a>
    
Get/set the antenna
Type
    
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Antenna" title="sionna.rt.Antenna">`Antenna`</a>




<em class="property">`property` </em>`array_size`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.AntennaArray.array_size" title="Permalink to this definition"></a>
    
Number of antennas in the array.
Dual-polarized antennas are counted as a single antenna.
Type
    
int (read-only)




<em class="property">`property` </em>`num_ant`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.AntennaArray.num_ant" title="Permalink to this definition"></a>
    
Number of linearly polarized antennas in the array.
Dual-polarized antennas are counted as two linearly polarized
antennas.
Type
    
int (read-only)




<em class="property">`property` </em>`positions`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.AntennaArray.positions" title="Permalink to this definition"></a>
    
Get/set  array of relative positions
$(x,y,z)$ [m] of each antenna (dual-polarized antennas are
counted as a single antenna and share the same position).
Type
    
[array_size, 3], <cite>tf.float</cite>




`rotated_positions`(<em class="sig-param">`orientation`</em>)<a class="reference internal" href="../_modules/sionna/rt/antenna_array.html#AntennaArray.rotated_positions">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.AntennaArray.rotated_positions" title="Permalink to this definition"></a>
    
Get the antenna positions rotated according to `orientation`
Input
    
**orientation** (<em>[3], tf.float</em>) – Orientation $(\alpha, \beta, \gamma)$ [rad] specified
through three angles corresponding to a 3D rotation
as defined in <a class="reference internal" href="../em_primer.html#equation-rotation">(3)</a>.

Output
    
<em>[array_size, 3]</em> – Rotated positions




### PlanarArray<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#planararray" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.rt.``PlanarArray`(<em class="sig-param">`num_rows`</em>, <em class="sig-param">`num_cols`</em>, <em class="sig-param">`vertical_spacing`</em>, <em class="sig-param">`horizontal_spacing`</em>, <em class="sig-param">`pattern`</em>, <em class="sig-param">`polarization``=``None`</em>, <em class="sig-param">`polarization_model``=``2`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/rt/antenna_array.html#PlanarArray">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.PlanarArray" title="Permalink to this definition"></a>
    
Class implementing a planar antenna array
    
The antennas are regularly spaced, located in the y-z plane, and
numbered column-first from the top-left to bottom-right corner.
Parameters
 
- **num_rows** (<em>int</em>) – Number of rows
- **num_cols** (<em>int</em>) – Number of columns
- **vertical_spacing** (<em>float</em>) – Vertical antenna spacing [multiples of wavelength].
- **horizontal_spacing** (<em>float</em>) – Horizontal antenna spacing [multiples of wavelength].
- **pattern** (<em>str</em><em>, </em><em>callable</em><em>, or </em><em>length-2 sequence of callables</em>) – Antenna pattern. Either one of
[“iso”, “dipole”, “hw_dipole”, “tr38901”],
or a callable, or a length-2 sequence of callables defining
antenna patterns. In the latter case, the antennas are dual
polarized and each callable defines the antenna pattern
in one of the two orthogonal polarization directions.
An antenna pattern is a callable that takes as inputs vectors of
zenith and azimuth angles of the same length and returns for each
pair the corresponding zenith and azimuth patterns. See <a class="reference internal" href="../em_primer.html#equation-c">(14)</a> for
more detail.
- **polarization** (<em>str</em><em> or </em><em>None</em>) – Type of polarization. For single polarization, must be “V” (vertical)
or “H” (horizontal). For dual polarization, must be “VH” or “cross”.
Only needed if `pattern` is a string.
- **polarization_model** (<em>int</em><em>, </em><em>one of</em><em> [</em><em>1</em><em>,</em><em>2</em><em>]</em>) – Polarization model to be used. Options <cite>1</cite> and <cite>2</cite>
refer to <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.polarization_model_1" title="sionna.rt.antenna.polarization_model_1">`polarization_model_1()`</a>
and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.polarization_model_2" title="sionna.rt.antenna.polarization_model_2">`polarization_model_2()`</a>,
respectively.
Defaults to <cite>2</cite>.
- **dtype** (<em>tf.complex64</em><em> or </em><em>tf.complex128</em>) – Datatype used for all computations.
Defaults to <cite>tf.complex64</cite>.



<p class="rubric">Example
```python
array = PlanarArray(8,4, 0.5, 0.5, "tr38901", "VH")
array.show()
```

<a class="reference internal image-reference" href="../_images/antenna_array.png"><img alt="../_images/antenna_array.png" src="https://nvlabs.github.io/sionna/_images/antenna_array.png" style="width: 640.0px; height: 480.0px;" /></a>

<em class="property">`property` </em>`antenna`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.PlanarArray.antenna" title="Permalink to this definition"></a>
    
Get/set the antenna
Type
    
<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Antenna" title="sionna.rt.Antenna">`Antenna`</a>




<em class="property">`property` </em>`array_size`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.PlanarArray.array_size" title="Permalink to this definition"></a>
    
Number of antennas in the array.
Dual-polarized antennas are counted as a single antenna.
Type
    
int (read-only)




<em class="property">`property` </em>`num_ant`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.PlanarArray.num_ant" title="Permalink to this definition"></a>
    
Number of linearly polarized antennas in the array.
Dual-polarized antennas are counted as two linearly polarized
antennas.
Type
    
int (read-only)




<em class="property">`property` </em>`positions`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.PlanarArray.positions" title="Permalink to this definition"></a>
    
Get/set  array of relative positions
$(x,y,z)$ [m] of each antenna (dual-polarized antennas are
counted as a single antenna and share the same position).
Type
    
[array_size, 3], <cite>tf.float</cite>




`rotated_positions`(<em class="sig-param">`orientation`</em>)<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.PlanarArray.rotated_positions" title="Permalink to this definition"></a>
    
Get the antenna positions rotated according to `orientation`
Input
    
**orientation** (<em>[3], tf.float</em>) – Orientation $(\alpha, \beta, \gamma)$ [rad] specified
through three angles corresponding to a 3D rotation
as defined in <a class="reference internal" href="../em_primer.html#equation-rotation">(3)</a>.

Output
    
<em>[array_size, 3]</em> – Rotated positions




`show`()<a class="reference internal" href="../_modules/sionna/rt/antenna_array.html#PlanarArray.show">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.PlanarArray.show" title="Permalink to this definition"></a>
    
Visualizes the antenna array
    
Antennas are depicted by markers that are annotated with the antenna
number. The marker is not related to the polarization of an antenna.
Output
    
`matplotlib.pyplot.Figure` – Figure depicting the antenna array




## Antennas<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#antennas" title="Permalink to this headline"></a>
    
We refer the user to the section “<a class="reference internal" href="../em_primer.html#far-field">Far Field of a Transmitting Antenna</a>” for various useful definitions and background on antenna modeling.
An <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Antenna" title="sionna.rt.Antenna">`Antenna`</a> can be single- or dual-polarized and has for each polarization direction a possibly different antenna pattern.
    
An antenna pattern is defined as a function $f:(\theta,\varphi)\mapsto (C_\theta(\theta, \varphi), C_\varphi(\theta, \varphi))$
that maps a pair of zenith and azimuth angles to zenith and azimuth pattern values.
You can easily define your own pattern or use one of the predefined <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#patterns">patterns</a> below.
    
Transmitters (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Transmitter" title="sionna.rt.Transmitter">`Transmitter`</a>) and receivers (<a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Receiver" title="sionna.rt.Receiver">`Receiver`</a>) are not equipped with an <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Antenna" title="sionna.rt.Antenna">`Antenna`</a> but an <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.AntennaArray" title="sionna.rt.AntennaArray">`AntennaArray`</a> that is composed of one or more antennas. All transmitters in a scene share the same <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.AntennaArray" title="sionna.rt.AntennaArray">`AntennaArray`</a> which can be set through the scene property <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.tx_array" title="sionna.rt.Scene.tx_array">`tx_array`</a>. The same holds for all receivers whose <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.AntennaArray" title="sionna.rt.AntennaArray">`AntennaArray`</a> can be set through <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Scene.rx_array" title="sionna.rt.Scene.rx_array">`rx_array`</a>.

### Antenna<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#antenna" title="Permalink to this headline"></a>

<em class="property">`class` </em>`sionna.rt.``Antenna`(<em class="sig-param">`pattern`</em>, <em class="sig-param">`polarization``=``None`</em>, <em class="sig-param">`polarization_model``=``2`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/rt/antenna.html#Antenna">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Antenna" title="Permalink to this definition"></a>
    
Class implementing an antenna
    
Creates an antenna object with an either predefined or custom antenna
pattern. Can be single or dual polarized.
Parameters
 
- **pattern** (<em>str</em><em>, </em><em>callable</em><em>, or </em><em>length-2 sequence of callables</em>) – Antenna pattern. Either one of
[“iso”, “dipole”, “hw_dipole”, “tr38901”],
or a callable, or a length-2 sequence of callables defining
antenna patterns. In the latter case, the antenna is dual
polarized and each callable defines the antenna pattern
in one of the two orthogonal polarization directions.
An antenna pattern is a callable that takes as inputs vectors of
zenith and azimuth angles of the same length and returns for each
pair the corresponding zenith and azimuth patterns.
- **polarization** (<em>str</em><em> or </em><em>None</em>) – Type of polarization. For single polarization, must be “V” (vertical)
or “H” (horizontal). For dual polarization, must be “VH” or “cross”.
Only needed if `pattern` is a string.
- **polarization_model** (<em>int</em><em>, </em><em>one of</em><em> [</em><em>1</em><em>,</em><em>2</em><em>]</em>) – Polarization model to be used. Options <cite>1</cite> and <cite>2</cite>
refer to <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.polarization_model_1" title="sionna.rt.antenna.polarization_model_1">`polarization_model_1()`</a>
and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.polarization_model_2" title="sionna.rt.antenna.polarization_model_2">`polarization_model_2()`</a>,
respectively.
Defaults to <cite>2</cite>.
- **dtype** (<em>tf.complex64</em><em> or </em><em>tf.complex128</em>) – Datatype used for all computations.
Defaults to <cite>tf.complex64</cite>.



<p class="rubric">Example
```python
>>> Antenna("tr38901", "VH")
```
<em class="property">`property` </em>`patterns`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.Antenna.patterns" title="Permalink to this definition"></a>
    
Antenna patterns for one or two
polarization directions
Type
    
<cite>list</cite>, <cite>callable</cite>




### compute_gain<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#compute-gain" title="Permalink to this headline"></a>

`sionna.rt.antenna.``compute_gain`(<em class="sig-param">`pattern`</em>)<a class="reference internal" href="../_modules/sionna/rt/antenna.html#compute_gain">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.compute_gain" title="Permalink to this definition"></a>
    
Computes the directivity, gain, and radiation efficiency of an antenna pattern
    
Given a function $f:(\theta,\varphi)\mapsto (C_\theta(\theta, \varphi), C_\varphi(\theta, \varphi))$
describing an antenna pattern <a class="reference internal" href="../em_primer.html#equation-c">(14)</a>, this function computes the gain $G$,
directivity $D$, and radiation efficiency $\eta_\text{rad}=G/D$
(see <a class="reference internal" href="../em_primer.html#equation-g">(12)</a> and text below).
Input
    
**pattern** (<em>callable</em>) – A callable that takes as inputs vectors of zenith and azimuth angles of the same
length and returns for each pair the corresponding zenith and azimuth patterns.

Output
 
- **D** (<em>float</em>) – Directivity $D$
- **G** (<em>float</em>) – Gain $G$
- **eta_rad** (<em>float</em>) – Radiation efficiency $\eta_\text{rad}$



<p class="rubric">Examples
```python
>>> compute_gain(tr38901_pattern)
(<tf.Tensor: shape=(), dtype=float32, numpy=9.606758>,
 <tf.Tensor: shape=(), dtype=float32, numpy=6.3095527>,
 <tf.Tensor: shape=(), dtype=float32, numpy=0.65678275>)
```


### visualize<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#visualize" title="Permalink to this headline"></a>

`sionna.rt.antenna.``visualize`(<em class="sig-param">`pattern`</em>)<a class="reference internal" href="../_modules/sionna/rt/antenna.html#visualize">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.visualize" title="Permalink to this definition"></a>
    
Visualizes an antenna pattern
    
This function visualizes an antenna pattern with the help of three
figures showing the vertical and horizontal cuts as well as a
three-dimensional visualization of the antenna gain.
Input
    
**pattern** (<em>callable</em>) – A callable that takes as inputs vectors of zenith and azimuth angles
of the same length and returns for each pair the corresponding zenith
and azimuth patterns.

Output
 
- `matplotlib.pyplot.Figure` – Vertical cut of the antenna gain
- `matplotlib.pyplot.Figure` – Horizontal cut of the antenna gain
- `matplotlib.pyplot.Figure` – 3D visualization of the antenna gain



<p class="rubric">Examples
```python
>>> fig_v, fig_h, fig_3d = visualize(hw_dipole_pattern)
```

<a class="reference internal image-reference" href="../_images/pattern_vertical.png"><img alt="../_images/pattern_vertical.png" src="https://nvlabs.github.io/sionna/_images/pattern_vertical.png" style="width: 512.0px; height: 384.0px;" /></a>

<a class="reference internal image-reference" href="../_images/pattern_horizontal.png"><img alt="../_images/pattern_horizontal.png" src="https://nvlabs.github.io/sionna/_images/pattern_horizontal.png" style="width: 512.0px; height: 384.0px;" /></a>

<a class="reference internal image-reference" href="../_images/pattern_3d.png"><img alt="../_images/pattern_3d.png" src="https://nvlabs.github.io/sionna/_images/pattern_3d.png" style="width: 512.0px; height: 384.0px;" /></a>

### dipole_pattern<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#dipole-pattern" title="Permalink to this headline"></a>

`sionna.rt.antenna.``dipole_pattern`(<em class="sig-param">`theta`</em>, <em class="sig-param">`phi`</em>, <em class="sig-param">`slant_angle``=``0.0`</em>, <em class="sig-param">`polarization_model``=``2`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/rt/antenna.html#dipole_pattern">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.dipole_pattern" title="Permalink to this definition"></a>
    
Short dipole pattern with linear polarizarion (Eq. 4-26a) <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#balanis97" id="id21">[Balanis97]</a>
Input
 
- **theta** (<em>array_like, float</em>) – Zenith angles wrapped within [0,pi] [rad]
- **phi** (<em>array_like, float</em>) – Azimuth angles wrapped within [-pi, pi) [rad]
- **slant_angle** (<em>float</em>) – Slant angle of the linear polarization [rad].
A slant angle of zero means vertical polarization.
- **polarization_model** (<em>int, one of [1,2]</em>) – Polarization model to be used. Options <cite>1</cite> and <cite>2</cite>
refer to <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.polarization_model_1" title="sionna.rt.antenna.polarization_model_1">`polarization_model_1()`</a>
and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.polarization_model_2" title="sionna.rt.antenna.polarization_model_2">`polarization_model_2()`</a>,
respectively.
Defaults to <cite>2</cite>.
- **dtype** (<em>tf.complex64 or tf.complex128</em>) – Datatype.
Defaults to <cite>tf.complex64</cite>.


Output
 
- **c_theta** (<em>array_like, complex</em>) – Zenith pattern
- **c_phi** (<em>array_like, complex</em>) – Azimuth pattern



<img alt="../_images/dipole_pattern.png" src="https://nvlabs.github.io/sionna/_images/dipole_pattern.png" />

### hw_dipole_pattern<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#hw-dipole-pattern" title="Permalink to this headline"></a>

`sionna.rt.antenna.``hw_dipole_pattern`(<em class="sig-param">`theta`</em>, <em class="sig-param">`phi`</em>, <em class="sig-param">`slant_angle``=``0.0`</em>, <em class="sig-param">`polarization_model``=``2`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/rt/antenna.html#hw_dipole_pattern">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.hw_dipole_pattern" title="Permalink to this definition"></a>
    
Half-wavelength dipole pattern with linear polarizarion (Eq. 4-84) <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#balanis97" id="id22">[Balanis97]</a>
Input
 
- **theta** (<em>array_like, float</em>) – Zenith angles wrapped within [0,pi] [rad]
- **phi** (<em>array_like, float</em>) – Azimuth angles wrapped within [-pi, pi) [rad]
- **slant_angle** (<em>float</em>) – Slant angle of the linear polarization [rad].
A slant angle of zero means vertical polarization.
- **polarization_model** (<em>int, one of [1,2]</em>) – Polarization model to be used. Options <cite>1</cite> and <cite>2</cite>
refer to <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.polarization_model_1" title="sionna.rt.antenna.polarization_model_1">`polarization_model_1()`</a>
and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.polarization_model_2" title="sionna.rt.antenna.polarization_model_2">`polarization_model_2()`</a>,
respectively.
Defaults to <cite>2</cite>.
- **dtype** (<em>tf.complex64 or tf.complex128</em>) – Datatype.
Defaults to <cite>tf.complex64</cite>.


Output
 
- **c_theta** (<em>array_like, complex</em>) – Zenith pattern
- **c_phi** (<em>array_like, complex</em>) – Azimuth pattern



<img alt="../_images/hw_dipole_pattern.png" src="https://nvlabs.github.io/sionna/_images/hw_dipole_pattern.png" />

### iso_pattern<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#iso-pattern" title="Permalink to this headline"></a>

`sionna.rt.antenna.``iso_pattern`(<em class="sig-param">`theta`</em>, <em class="sig-param">`phi`</em>, <em class="sig-param">`slant_angle``=``0.0`</em>, <em class="sig-param">`polarization_model``=``2`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/rt/antenna.html#iso_pattern">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.iso_pattern" title="Permalink to this definition"></a>
    
Isotropic antenna pattern with linear polarizarion
Input
 
- **theta** (<em>array_like, float</em>) – Zenith angles wrapped within [0,pi] [rad]
- **phi** (<em>array_like, float</em>) – Azimuth angles wrapped within [-pi, pi) [rad]
- **slant_angle** (<em>float</em>) – Slant angle of the linear polarization [rad].
A slant angle of zero means vertical polarization.
- **polarization_model** (<em>int, one of [1,2]</em>) – Polarization model to be used. Options <cite>1</cite> and <cite>2</cite>
refer to <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.polarization_model_1" title="sionna.rt.antenna.polarization_model_1">`polarization_model_1()`</a>
and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.polarization_model_2" title="sionna.rt.antenna.polarization_model_2">`polarization_model_2()`</a>,
respectively.
Defaults to <cite>2</cite>.
- **dtype** (<em>tf.complex64 or tf.complex128</em>) – Datatype.
Defaults to <cite>tf.complex64</cite>.


Output
 
- **c_theta** (<em>array_like, complex</em>) – Zenith pattern
- **c_phi** (<em>array_like, complex</em>) – Azimuth pattern



<img alt="../_images/iso_pattern.png" src="https://nvlabs.github.io/sionna/_images/iso_pattern.png" />

### tr38901_pattern<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#tr38901-pattern" title="Permalink to this headline"></a>

`sionna.rt.antenna.``tr38901_pattern`(<em class="sig-param">`theta`</em>, <em class="sig-param">`phi`</em>, <em class="sig-param">`slant_angle``=``0.0`</em>, <em class="sig-param">`polarization_model``=``2`</em>, <em class="sig-param">`dtype``=``tf.complex64`</em>)<a class="reference internal" href="../_modules/sionna/rt/antenna.html#tr38901_pattern">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.tr38901_pattern" title="Permalink to this definition"></a>
    
Antenna pattern from 3GPP TR 38.901 (Table 7.3-1) <a class="reference internal" href="channel.wireless.html#tr38901" id="id23">[TR38901]</a>
Input
 
- **theta** (<em>array_like, float</em>) – Zenith angles wrapped within [0,pi] [rad]
- **phi** (<em>array_like, float</em>) – Azimuth angles wrapped within [-pi, pi) [rad]
- **slant_angle** (<em>float</em>) – Slant angle of the linear polarization [rad].
A slant angle of zero means vertical polarization.
- **polarization_model** (<em>int, one of [1,2]</em>) – Polarization model to be used. Options <cite>1</cite> and <cite>2</cite>
refer to <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.polarization_model_1" title="sionna.rt.antenna.polarization_model_1">`polarization_model_1()`</a>
and <a class="reference internal" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.polarization_model_2" title="sionna.rt.antenna.polarization_model_2">`polarization_model_2()`</a>,
respectively.
Defaults to <cite>2</cite>.
- **dtype** (<em>tf.complex64 or tf.complex128</em>) – Datatype.
Defaults to <cite>tf.complex64</cite>.


Output
 
- **c_theta** (<em>array_like, complex</em>) – Zenith pattern
- **c_phi** (<em>array_like, complex</em>) – Azimuth pattern



<img alt="../_images/tr38901_pattern.png" src="https://nvlabs.github.io/sionna/_images/tr38901_pattern.png" />

### polarization_model_1<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#polarization-model-1" title="Permalink to this headline"></a>

`sionna.rt.antenna.``polarization_model_1`(<em class="sig-param">`c_theta`</em>, <em class="sig-param">`theta`</em>, <em class="sig-param">`phi`</em>, <em class="sig-param">`slant_angle`</em>)<a class="reference internal" href="../_modules/sionna/rt/antenna.html#polarization_model_1">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.polarization_model_1" title="Permalink to this definition"></a>
    
Model-1 for polarized antennas from 3GPP TR 38.901
    
Transforms a vertically polarized antenna pattern $\tilde{C}_\theta(\theta, \varphi)$
into a linearly polarized pattern whose direction
is specified by a slant angle $\zeta$. For example,
$\zeta=0$ and $\zeta=\pi/2$ correspond
to vertical and horizontal polarization, respectively,
and $\zeta=\pm \pi/4$ to a pair of cross polarized
antenna elements.
    
The transformed antenna pattern is given by (7.3-3) <a class="reference internal" href="channel.wireless.html#tr38901" id="id24">[TR38901]</a>:

$$
\begin{split}\begin{align}
    \begin{bmatrix}
        C_\theta(\theta, \varphi) \\
        C_\varphi(\theta, \varphi)
    \end{bmatrix} &= \begin{bmatrix}
     \cos(\psi) \\
     \sin(\psi)
    \end{bmatrix} \tilde{C}_\theta(\theta, \varphi)\\
    \cos(\psi) &= \frac{\cos(\zeta)\sin(\theta)+\sin(\zeta)\sin(\varphi)\cos(\theta)}{\sqrt{1-\left(\cos(\zeta)\cos(\theta)-\sin(\zeta)\sin(\varphi)\sin(\theta)\right)^2}} \\
    \sin(\psi) &= \frac{\sin(\zeta)\cos(\varphi)}{\sqrt{1-\left(\cos(\zeta)\cos(\theta)-\sin(\zeta)\sin(\varphi)\sin(\theta)\right)^2}}
\end{align}\end{split}
$$

Input
 
- **c_tilde_theta** (<em>array_like, complex</em>) – Zenith pattern
- **theta** (<em>array_like, float</em>) – Zenith angles wrapped within [0,pi] [rad]
- **phi** (<em>array_like, float</em>) – Azimuth angles wrapped within [-pi, pi) [rad]
- **slant_angle** (<em>float</em>) – Slant angle of the linear polarization [rad].
A slant angle of zero means vertical polarization.


Output
 
- **c_theta** (<em>array_like, complex</em>) – Zenith pattern
- **c_phi** (<em>array_like, complex</em>) – Azimuth pattern




### polarization_model_2<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#polarization-model-2" title="Permalink to this headline"></a>

`sionna.rt.antenna.``polarization_model_2`(<em class="sig-param">`c`</em>, <em class="sig-param">`slant_angle`</em>)<a class="reference internal" href="../_modules/sionna/rt/antenna.html#polarization_model_2">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.antenna.polarization_model_2" title="Permalink to this definition"></a>
    
Model-2 for polarized antennas from 3GPP TR 38.901
    
Transforms a vertically polarized antenna pattern $\tilde{C}_\theta(\theta, \varphi)$
into a linearly polarized pattern whose direction
is specified by a slant angle $\zeta$. For example,
$\zeta=0$ and $\zeta=\pi/2$ correspond
to vertical and horizontal polarization, respectively,
and $\zeta=\pm \pi/4$ to a pair of cross polarized
antenna elements.
    
The transformed antenna pattern is given by (7.3-4/5) <a class="reference internal" href="channel.wireless.html#tr38901" id="id25">[TR38901]</a>:

$$
\begin{split}\begin{align}
    \begin{bmatrix}
        C_\theta(\theta, \varphi) \\
        C_\varphi(\theta, \varphi)
    \end{bmatrix} &= \begin{bmatrix}
     \cos(\zeta) \\
     \sin(\zeta)
    \end{bmatrix} \tilde{C}_\theta(\theta, \varphi)
\end{align}\end{split}
$$

Input
 
- **c_tilde_theta** (<em>array_like, complex</em>) – Zenith pattern
- **slant_angle** (<em>float</em>) – Slant angle of the linear polarization [rad].
A slant angle of zero means vertical polarization.


Output
 
- **c_theta** (<em>array_like, complex</em>) – Zenith pattern
- **c_phi** (<em>array_like, complex</em>) – Azimuth pattern




## Utility Functions<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#utility-functions" title="Permalink to this headline"></a>

### cross<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#cross" title="Permalink to this headline"></a>

`sionna.rt.``cross`(<em class="sig-param">`u`</em>, <em class="sig-param">`v`</em>)<a class="reference internal" href="../_modules/sionna/rt/utils.html#cross">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.cross" title="Permalink to this definition"></a>
    
Computes the cross (or vector) product between u and v
Input
 
- **u** (<em>[…,3]</em>) – First vector
- **v** (<em>[…,3]</em>) – Second vector


Output
    
<em>[…,3]</em> – Cross product between `u` and `v`



### dot<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#dot" title="Permalink to this headline"></a>

`sionna.rt.``dot`(<em class="sig-param">`u`</em>, <em class="sig-param">`v`</em>, <em class="sig-param">`keepdim``=``False`</em>, <em class="sig-param">`clip``=``False`</em>)<a class="reference internal" href="../_modules/sionna/rt/utils.html#dot">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.dot" title="Permalink to this definition"></a>
    
Computes and the dot (or scalar) product between u and v
Input
 
- **u** (<em>[…,3]</em>) – First vector
- **v** (<em>[…,3]</em>) – Second vector
- **keepdim** (<em>bool</em>) – If <cite>True</cite>, keep the last dimension.
Defaults to <cite>False</cite>.
- **clip** (<em>bool</em>) – If <cite>True</cite>, clip output to [-1,1].
Defaults to <cite>False</cite>.


Output
    
<em>[…,1] or […]</em> – Dot product between `u` and `v`.
The last dimension is removed if `keepdim`
is set to <cite>False</cite>.



### normalize<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#normalize" title="Permalink to this headline"></a>

`sionna.rt.``normalize`(<em class="sig-param">`v`</em>)<a class="reference internal" href="../_modules/sionna/rt/utils.html#normalize">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.normalize" title="Permalink to this definition"></a>
    
Normalizes `v` to unit norm
Input
    
**v** (<em>[…,3], tf.float</em>) – Vector

Output
 
- <em>[…,3], tf.float</em> – Normalized vector
- <em>[…], tf.float</em> – Norm of the unnormalized vector




### phi_hat<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#phi-hat" title="Permalink to this headline"></a>

`sionna.rt.``phi_hat`(<em class="sig-param">`phi`</em>)<a class="reference internal" href="../_modules/sionna/rt/utils.html#phi_hat">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.phi_hat" title="Permalink to this definition"></a>
    
Computes the spherical unit vector
$\hat{\boldsymbol{\varphi}}(\theta, \varphi)$
as defined in <a class="reference internal" href="../em_primer.html#equation-spherical-vecs">(1)</a>
Input
    
**phi** (same shape as `theta`, tf.float) – Azimuth angles $\varphi$ [rad]

Output
    
**theta_hat** (`phi.shape` + [3], tf.float) – Vector $\hat{\boldsymbol{\varphi}}(\theta, \varphi)$



### rotate<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#rotate" title="Permalink to this headline"></a>

`sionna.rt.``rotate`(<em class="sig-param">`p`</em>, <em class="sig-param">`angles`</em>)<a class="reference internal" href="../_modules/sionna/rt/utils.html#rotate">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.rotate" title="Permalink to this definition"></a>
    
Rotates points `p` by the `angles` according
to the 3D rotation defined in <a class="reference internal" href="../em_primer.html#equation-rotation">(3)</a>
Input
 
- **p** (<em>[…,3], tf.float</em>) – Points to rotate
- **angles** (<em>[…, 3]</em>) – Angles for the rotations [rad].
The last dimension corresponds to the angles
$(\alpha,\beta,\gamma)$ that define
rotations about the axes $(z, y, x)$,
respectively.


Output
    
<em>[…,3]</em> – Rotated points `p`



### rotation_matrix<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#rotation-matrix" title="Permalink to this headline"></a>

`sionna.rt.``rotation_matrix`(<em class="sig-param">`angles`</em>)<a class="reference internal" href="../_modules/sionna/rt/utils.html#rotation_matrix">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.rotation_matrix" title="Permalink to this definition"></a>
    
Computes rotation matrices as defined in <a class="reference internal" href="../em_primer.html#equation-rotation">(3)</a>
    
The closed-form expression in (7.1-4) <a class="reference internal" href="channel.wireless.html#tr38901" id="id26">[TR38901]</a> is used.
Input
    
**angles** (<em>[…,3], tf.float</em>) – Angles for the rotations [rad].
The last dimension corresponds to the angles
$(\alpha,\beta,\gamma)$ that define
rotations about the axes $(z, y, x)$,
respectively.

Output
    
<em>[…,3,3], tf.float</em> – Rotation matrices



### rot_mat_from_unit_vecs<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#rot-mat-from-unit-vecs" title="Permalink to this headline"></a>

`sionna.rt.``rot_mat_from_unit_vecs`(<em class="sig-param">`a`</em>, <em class="sig-param">`b`</em>)<a class="reference internal" href="../_modules/sionna/rt/utils.html#rot_mat_from_unit_vecs">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.rot_mat_from_unit_vecs" title="Permalink to this definition"></a>
    
Computes Rodrigues` rotation formula <a class="reference internal" href="../em_primer.html#equation-rodrigues-matrix">(6)</a>
Input
 
- **a** (<em>[…,3], tf.float</em>) – First unit vector
- **b** (<em>[…,3], tf.float</em>) – Second unit vector


Output
    
<em>[…,3,3], tf.float</em> – Rodrigues’ rotation matrix



### r_hat<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#r-hat" title="Permalink to this headline"></a>

`sionna.rt.``r_hat`(<em class="sig-param">`theta`</em>, <em class="sig-param">`phi`</em>)<a class="reference internal" href="../_modules/sionna/rt/utils.html#r_hat">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.r_hat" title="Permalink to this definition"></a>
    
Computes the spherical unit vetor $\hat{\mathbf{r}}(\theta, \phi)$
as defined in <a class="reference internal" href="../em_primer.html#equation-spherical-vecs">(1)</a>
Input
 
- **theta** (<em>arbitrary shape, tf.float</em>) – Zenith angles $\theta$ [rad]
- **phi** (same shape as `theta`, tf.float) – Azimuth angles $\varphi$ [rad]


Output
    
**rho_hat** (`phi.shape` + [3], tf.float) – Vector $\hat{\mathbf{r}}(\theta, \phi)$  on unit sphere



### sample_points_on_hemisphere<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sample-points-on-hemisphere" title="Permalink to this headline"></a>

`sionna.rt.``sample_points_on_hemisphere`(<em class="sig-param">`normals`</em>, <em class="sig-param">`num_samples``=``1`</em>)<a class="reference internal" href="../_modules/sionna/rt/utils.html#sample_points_on_hemisphere">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.sample_points_on_hemisphere" title="Permalink to this definition"></a>
    
Randomly sample points on hemispheres defined by their normal vectors
Input
 
- **normals** (<em>[batch_size, 3], tf.float</em>) – Normal vectors defining hemispheres
- **num_samples** (<em>int</em>) – Number of random samples to draw for each hemisphere
defined by its normal vector.
Defaults to 1.


Output
    
**points** (<em>[batch_size, num_samples, 3], tf.float or [batch_size, 3], tf.float if num_samples=1.</em>) – Random points on the hemispheres



### theta_hat<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#theta-hat" title="Permalink to this headline"></a>

`sionna.rt.``theta_hat`(<em class="sig-param">`theta`</em>, <em class="sig-param">`phi`</em>)<a class="reference internal" href="../_modules/sionna/rt/utils.html#theta_hat">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.theta_hat" title="Permalink to this definition"></a>
    
Computes the spherical unit vector
$\hat{\boldsymbol{\theta}}(\theta, \varphi)$
as defined in <a class="reference internal" href="../em_primer.html#equation-spherical-vecs">(1)</a>
Input
 
- **theta** (<em>arbitrary shape, tf.float</em>) – Zenith angles $\theta$ [rad]
- **phi** (same shape as `theta`, tf.float) – Azimuth angles $\varphi$ [rad]


Output
    
**theta_hat** (`phi.shape` + [3], tf.float) – Vector $\hat{\boldsymbol{\theta}}(\theta, \varphi)$



### theta_phi_from_unit_vec<a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#theta-phi-from-unit-vec" title="Permalink to this headline"></a>

`sionna.rt.``theta_phi_from_unit_vec`(<em class="sig-param">`v`</em>)<a class="reference internal" href="../_modules/sionna/rt/utils.html#theta_phi_from_unit_vec">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/rt.html#sionna.rt.theta_phi_from_unit_vec" title="Permalink to this definition"></a>
    
Computes zenith and azimuth angles ($\theta,\varphi$)
from unit-norm vectors as described in <a class="reference internal" href="../em_primer.html#equation-theta-phi">(2)</a>
Input
    
**v** (<em>[…,3], tf.float</em>) – Tensor with unit-norm vectors in the last dimension

Output
 
- **theta** (<em>[…], tf.float</em>) – Zenith angles $\theta$
- **phi** (<em>[…], tf.float</em>) – Azimuth angles $\varphi$





References:
Balanis97(<a href="https://nvlabs.github.io/sionna/api/rt.html#id21">1</a>,<a href="https://nvlabs.github.io/sionna/api/rt.html#id22">2</a>)
<ol class="upperalpha simple">
- Balanis, “Antenna Theory: Analysis and Design,” 2nd Edition, John Wiley & Sons, 1997.
</ol>

ITUR_P2040_2(<a href="https://nvlabs.github.io/sionna/api/rt.html#id16">1</a>,<a href="https://nvlabs.github.io/sionna/api/rt.html#id17">2</a>)
    
ITU-R, “Effects of building materials and structures on radiowave propagation above about 100 MHz“, Recommendation ITU-R P.2040-2

<a class="fn-backref" href="https://nvlabs.github.io/sionna/api/rt.html#id2">SurfaceIntegral</a>
    
Wikipedia, “<a class="reference external" href="https://en.wikipedia.org/wiki/Surface_integral">Surface integral</a>”, accessed Jun. 22, 2023.



