from .cli import main

if __name__ == "__main__":  # con multiprocessing (spawn en macOS) los hijos reimportan este módulo
    raise SystemExit(main())
