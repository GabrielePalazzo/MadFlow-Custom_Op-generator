# jinjatest

Run the Python version with `python madflow_exec.py [options]`.
Run the Custom Operator version with `python cumadflow_exec.py [options]`.

Custom Operator files are generated inside the `gpu/` directory.

In order to generate a new Custom Operator, remove the prevous files with `make clean_all`.
Generate the matrix element file inside the `prov/` directory (`madflow --dry_run -o prov [options]`), then generate the Custom Op with `python custom_op_generator.py` and compile it with `make`.

`custom_op_generator.py` points to `wavefunctions.py` inside the madflow package:
```
file_sources = ['/home/palazzo/madflow/python_package/madflow/wavefunctions_flow.py']
```
