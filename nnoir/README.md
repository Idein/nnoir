# nnoir

## install

```
pip install .
```

## example

### create & save

```
inputs  = [nnoir.Value(b'v0', 'float', (10,10)),
           nnoir.Value(b'v1', 'float', (10,10))]
outputs = [nnoir.Value(b'v2', 'float', (10,10))]
nodes = inputs + outputs
input_names = [ x.name for x in inputs ]
output_names = [ x.name for x in outputs ]
functions = [nnoir.functions.Add(input_names, output_names)]
result = nnoir.NNOIR(b'Add', b'add_test', 0.1, input_names, output_names, nodes, functions)
result.dump('add.nnoir')
```

### load

```
add_nnoir = nnoir.load('add.nnoir')
```
