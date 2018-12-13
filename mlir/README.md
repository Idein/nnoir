# mlir

## install
` pip install .`

## example
### create&save
```
    inputs  = [mlir.Node('v0', 'float', (10,10)),
               mlir.Node('v1', 'float', (10,10))]
    outputs = [mlir.Node('v2', 'float', (10,10))]
    nodes = inputs + outputs
    input_names = [ x.name for x in inputs ]
    output_names = [ x.name for x in outputs ]
    function = mlir.edges.Add(input_names, output_names)
    result = mlir.MLIR('Add', 'mlir2chainer_test', 0.1, input_names, output_names, nodes, [function])
    result.dump('add.mlir')
```
### load
```
    add_mlir = mlir.load('add.mlir')
```
