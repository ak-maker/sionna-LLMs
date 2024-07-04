
# Configuration<a class="headerlink" href="https://nvlabs.github.io/sionna/api/config.html#configuration" title="Permalink to this headline"></a>
    
Sionna’s configuration API. It can be used to set global variables which can be used
by all modules and functions.

<em class="property">`class` </em>`sionna.``Config`<a class="reference internal" href="../_modules/sionna/config.html#Config">`[source]`</a><a class="headerlink" href="https://nvlabs.github.io/sionna/api/config.html#sionna.Config" title="Permalink to this definition"></a>
    
The Sionna configuration class.
    
This class is used to define global configuration variables
that can be accessed from all modules and functions. It
is instantiated in `sionna.__init__()` and its properties can be
accessed as `sionna.config.desired_property`.

<em class="property">`property` </em>`xla_compat`<a class="headerlink" href="https://nvlabs.github.io/sionna/api/config.html#sionna.Config.xla_compat" title="Permalink to this definition"></a>
    
Ensure that functions execute in an XLA compatible way.
    
Not all TensorFlow ops support the three execution modes for
all dtypes: Eager, Graph, and Graph with XLA. For this reason,
some functions are implemented differently depending on the
execution mode. As it is currently impossible to programmatically
determine if a function is executed in Graph or Graph with XLA mode,
the `xla_compat` property can be used to indicate which execution
mode is desired. Note that most functions will work in all execution
modes independently of the value of this property.
    
This property can be used like this:
```python
import sionna
sionna.config.xla_compat=True
@tf.function(jit_compile=True)
def func()
    # Implementation
func()
```

Type
    
bool



