# mlir

## install

```
pip install .
```

## example

### create & save

```
inputs  = [mlir.Value(b'v0', 'float', (10,10)),
           mlir.Value(b'v1', 'float', (10,10))]
outputs = [mlir.Value(b'v2', 'float', (10,10))]
nodes = inputs + outputs
input_names = [ x.name for x in inputs ]
output_names = [ x.name for x in outputs ]
functions = [mlir.functions.Add(input_names, output_names)]
result = mlir.MLIR(b'Add', b'add_test', 0.1, input_names, output_names, nodes, functions)
result.dump('add.mlir')
```

### load

```
add_mlir = mlir.load('add.mlir')
```
