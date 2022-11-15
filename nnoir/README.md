# NNOIR

## Install

```
pip install nnoir
```

## Example

### Create & Save

```python
inputs  = [nnoir.Value(b'v0', dtype='<f4', shape=(10,10)),
           nnoir.Value(b'v1', dtype='<f4', shape=(10,10))]
outputs = [nnoir.Value(b'v2', dtype='<f4', shape=(10,10))]
nodes = inputs + outputs
input_names = [ x.name for x in inputs ]
output_names = [ x.name for x in outputs ]
functions = [nnoir.functions.Add(input_names, output_names)]
result = nnoir.NNOIR(b'Add', b'add_test', '0.1', input_names, output_names, nodes, functions)
result.dump('add.nnoir')
```

### Load

```python
add_nnoir = nnoir.load('add.nnoir')
```

### Read/Write metadata from command line

```bash
$ nnoir-metadata mobilenet_v2.nnoir
name = MobileNet_v2
description =
generator.name = chainer
generator.version = 7.7.0
$ nnoir-metadata mobilenet_v2.nnoir description "This is Mobilet V2 (description added by nnoir-metadata)"
$ nnoir-metadata mobilenet_v2.nnoir
name = MobileNet_v2
description = This is Mobilet V2 (description added by nnoir-metadata)
generator.name = chainer
generator.version = 7.7.0
```