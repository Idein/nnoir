import nnoir

if __name__ == '__main__':
    inputs  = [nnoir.Value(b'v0', dtype='<f4', shape=(1, 3, 4, 5))]
    outputs = [nnoir.Value(b'v2', dtype='<f4', shape=(1, 3, 4, 5))]
    nodes = inputs + outputs
    input_names = [x.name for x in inputs]
    output_names = [x.name for x in outputs]
    functions = [nnoir.functions.Erf(input_names, output_names)]
    result = nnoir.NNOIR(
        b"Erf",
        b"erf_test",
        b"0.1",
        input_names,
        output_names,
        nodes,
        functions,
    )
    result.dump('Erf.nnoir')