def run_hybrid(input_path, output, config):
    from .dynamic import run_dynamic
    run_dynamic(input_path, output, config)
