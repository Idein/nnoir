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
$ nnoir-metadata resnet_50.nnoir
name = CaffeFunction
description =
generator.name = chainer
generator.version = 7.7.0
$ nnoir-metadata resnet_50.nnoir --write-description "This is resnet_50 (written by nnoir-metada)"
$ nnoir-metadata resnet_50.nnoir                                            
name = CaffeFunction
description = This is resnet_50 (written by nnoir-metada)
generator.name = chainer
generator.version = 7.7.0
$ nnoir-metadata resnet_50.nnoir --write-name "CaffeFunction_V2"
$ nnoir-metadata resnet_50.nnoir
name = CaffeFunction_V2
description = This is resnet_50 (written by nnoir-metada)
generator.name = chainer
generator.version = 7.7.0
```