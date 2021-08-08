# tf.config.list_physical_devices

Return a list of physical devices visible to the host runtime.

Physical devices are hardware devices present on the host machine. By default all discovered CPU and GPU devices are considered visible.

This API allows querying the physical hardware resources prior to runtime initialization. Thus, giving an opportunity to call any additional configuration APIs. This is in contrast to tf.config.list_logical_devices, which triggers runtime initialization in order to list the configured devices.

The following example lists the number of visible GPUs on the host.

```
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
```

However, the number of GPUs available to the runtime may change during runtime initialization due to marking certain devices as not visible or configuring multiple logical devices.

# tf.config.set_visible_devices

Set the list of visible devices.

Specifies which PhysicalDevice objects are visible to the runtime. TensorFlow will only allocate memory and place operations on visible physical devices, as otherwise no LogicalDevice will be created on them. By default all discovered devices are marked as visible.

The following example demonstrates disabling the first GPU on the machine

```
physical_devices = tf.config.list_physical_devices('GPU')
try:
  # Disable first GPU
  tf.config.set_visible_devices(physical_devices[1:], 'GPU')
  logical_devices = tf.config.list_logical_devices('GPU')
  # Logical device was not created for first GPU
  assert len(logical_devices) == len(physical_devices) - 1
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass
```

# tf.config.list_logical_devices

Return a list of logical devices created by runtime

Logical devices may correspond to physical devices or remote devices in the cluster. Operations and tensors may be placed on these devices by using the name of the tf.config.LogicalDevice.

Calling tf.config.list_logical_devices triggers the runtime to configure any tf.config.PhysicalDevice visible to the runtime, thereby preventing further configuration. To avoid runtime initialization, call tf.config.list_physical_devices instead.

```
logical_devices = tf.config.list_logical_devices('GPU')
if len(logical_devices) > 0:
  # Allocate on GPU:0
  with tf.device(logical_devices[0].name):
    one = tf.constant(1)
  # Allocate on GPU:1
  with tf.device(logical_devices[1].name):
    two = tf.constant(2)
```

